import lightkurve as lk
import matplotlib.pyplot as plt
import dill
import numpy as np
import batman
from exoplanet import phase_fold
# import pandas
# import my_constants as myc
# from PyAstronomy import pyasl
# from scipy import interpolate
# from scipy.signal import medfilt
# from astropy.io import fits
from astropy.stats import sigma_clip
# import dave.vetting.ModShift as ms
import dave.vetting.RoboVet as rv
from subprocess import check_output

import seaborn as sns
sns.set()


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    st_r, st_m = 0.456, 0.497
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


run = "../save_K2SFF_16_300_2000_1000"; lc_file = "LC_K2SFF.dat"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)

rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
sas = keplerslaw(pers)

# t, f, e = np.loadtxt(lc_file, unpack=True, delimiter=",")
t, f, e = data.LCtime, data.LC, data.LCerror

# d = fits.open("hlsp_k2sff_k2_lightcurve_247887989-c13_kepler_v1_llc.fits")
# dat = d[1].data    # T, FRAW, FCOR, ARCLENGTH, MOVING, CADENCENO
# t = dat["T"]
# f = dat["FCOR"]
# t_u = np.linspace(t[0], t[-1], int((t[-1]-t[0])/0.020432106))   # uniform spacing
# mf = medfilt(f, 25)     # median filter
# cv = interpolate.interp1d(t, mf)(t_u)   # interpolate trend to full LC
# gc = pyasl.broadGaussFast(t_u, cv, 0.05, edgeHandling="firstlast")  # gaussian convolve to smooth
# gc = interpolate.interp1d(t_u, gc)(t)   # interpolate back to only data points from K2SFF

# plt.plot(t, f, ".")
# plt.plot(t, gc)
# _ = [plt.axvline(t0s[3]+i*pers[3], c="k") for i in range(3)]
# plt.show()

# f = f - gc + 1.    # correct LC

models = []     # best-fit model for each planet
pars = []

for i in range(4):
    rp, inc, per, t0, a = rps[i], incs[i], pers[i], t0s[i], sas[i]      # best-fit values for planet

    # create best-fit model
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.
    params.w = 90.
    params.u = [0.5079, 0.2239]
    params.limb_dark = "quadratic"

    p1, fp = phase_fold(t, f, per, t0+per/4.)   # phase-folded flux

    # model matching t
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.0)
    m = bat.light_curve(params)
    # p2, mp = phase_fold(t, m, per, t0+per/4.)   # phase-folded model

    np.savetxt("pl_{}.dat".format(["b", "c", "d", "01"][i]), np.c_[t, f, m])

    models.append(np.asarray(m))
    pars.append(params)


models = np.asarray(models)
m_tot = np.sum(models[:3], axis=0) - 2.     # total transit model for system

f01 = f - m_tot + 1.

res = f01 - models[3]   # residuals
cut = sigma_clip(f01, sigma_lower=8., sigma_upper=3.).mask
mask = cut == 0
# t, f01, res = t[mask], f01[mask], res[mask]

#
#

fnone = f-np.sum(models, axis=0)+4.
np.savetxt("fnone.dat", np.array([t, fnone]).T, delimiter=",")

plt.plot(t, fnone, ".")
plt.show()

np.savetxt("pl_f01.dat", np.c_[t, f01, models[3]])


def runModShift(fname, plotname, objectname, period, epoch, modplotint):
    """Run the Model-Shift test
    Inputs:
    -------------
    time
        The array of time values days.
    flux
        The array of observed fluxes correspodnding to each time.
    model
        The array of model fluxes corresponding to each time.
    plotname
        The name for the output plot
    objectname
        The name of the object, to be displayed in the plot title
    period
        The period of the system in days.
    epoch
        The epoch of the system in days.
    modplotint
       If modplotint==1, then plot will be produced, else it won't


    Returns:
    -------------
    A dictionary containing the following keys:
    mod_sig_pri
      The significance of the primary event assuming white noise
    mod_sig_sec
      The significance of the secondary event assuming white noise
    mod_sig_ter
      The significance of the tertiary event assuming white noise
    mod_sig_pos
      The significance of the positive event assuming white noise
    mod_sig_oe
      The significance of the odd-even metric
    mod_dmm
      The ratio of the individual depths's median and mean values.
    mod_shape
      The shape metric.
    mod_sig_fa1
      The False Alarm threshold assuming 20,000 objects evaluated
    mod_sig_fa2
      The False Alarm threshold for two events within the phased light curve
    mod_Fred
      The ratio of the red noise to the white noise in the phased light curve at the transit timescale
    mod_ph_pri
      The phase of the primary event
    mod_ph_sec
      The phase of the secondary event
    mod_ph_sec
      The phase of the tertiary event
    mod_ph_pos
      The phase of the primary event
    mod_secdepth
      The depth of the secondary event
    mod_secdeptherr
      The error in the depth of the secondary event
    Output:
    ----------
    The model-shift plot is also created as a PDF file if plot==1
    """

    timeout_sec = 10
    objectname = objectname.replace(" ", "_")

    # Write data to a tempoary file so it can be read by model-shift
    # compiled C code. mkstemp ensures the file is written to a random
    # location so the code can be run in parallel.
    # fpNum, tmpFilename = tempfile.mkstemp(prefix="modshift-%s" % (objectname))
    # numpy.savetxt(tmpFilename, numpy.c_[time, flux, model])

    # Run modshift, and return the output
    # Python 2's subprocess module does not easily support timeouts, so
    # instead we use the shell's version
    # Insert rant here asking why subprocess doesn't have a timeout when it's
    # the complicated module that was supposed to handle communication better.

    cmd = ["/Users/rwells/Documents/dave/vetting/modshift",
           fname, plotname, objectname, str(period), str(epoch), str(modplotint)]

    modshiftcmdout = check_output(cmd)

    # Delete the input text file
    # os.close(fpNum)
    # os.remove(tmpFilename)

    # Read the modshift output back in to variables
    info = modshiftcmdout.split()
    del modshiftcmdout
    mod_sig_pri = float(info[1])
    mod_sig_sec = float(info[2])
    mod_sig_ter = float(info[3])
    mod_sig_pos = float(info[4])
    mod_sig_oe = float(info[5])
    mod_dmm = float(info[6])
    mod_shape = float(info[7])
    mod_sig_fa1 = float(info[8])
    mod_sig_fa2 = float(info[9])
    mod_Fred = float(info[10])
    mod_ph_pri = float(info[11])
    mod_ph_sec = float(info[12])
    mod_ph_ter = float(info[13])
    mod_ph_pos = float(info[14])
    mod_secdepth = float(info[15])
    mod_secdeptherr = float(info[16])

    return {'mod_sig_pri': mod_sig_pri, 'mod_sig_sec': mod_sig_sec, 'mod_sig_ter': mod_sig_ter,
            'mod_sig_pos': mod_sig_pos, 'mod_sig_oe': mod_sig_oe, 'mod_dmm': mod_dmm, 'mod_shape': mod_shape,
            'mod_sig_fa1': mod_sig_fa1, 'mod_sig_fa2': mod_sig_fa2, 'mod_Fred': mod_Fred, 'mod_ph_pri': mod_ph_pri,
            'mod_ph_sec': mod_ph_sec, 'mod_ph_ter': mod_ph_ter, 'mod_ph_pos': mod_ph_pos, 'mod_secdepth': mod_secdepth,
            'mod_secdeptherr': mod_secdeptherr}


for i in range(4):
    pl = ["b", "c", "d", "f01"][i]
    ms_dict = runModShift("/Users/rwells/Desktop/followup-k2-133/transit/vetting/pl_{}.dat".format(pl),
                          pl, pl, pers[i], t0s[i], 1)

    print rv.roboVet(ms_dict)
