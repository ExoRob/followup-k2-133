import lightkurve as lk
import matplotlib.pyplot as plt
import dill
import numpy as np
import batman
from exoplanet import phase_fold
import pandas
import my_constants as myc
from PyAstronomy import pyasl
from scipy import interpolate
from scipy.signal import medfilt
from astropy.io import fits
from astropy.stats import sigma_clip
from pybls import BLS

import seaborn as sns
sns.set()


def calc_cdpp(y, win=4):
    """
    Return the scatter in ppm based on the median running standard deviation
    for a window size of :py:obj:`win` = 13 cadences (for K2, this
    is ~6.5 hours, as in VJ14).
    :param ndarray y: The array whose CDPP is to be computed
    :param int win: The window size in cadences. Default `13`
    """

    def Chunks(l, n, all_=False):
        """
        Returns a generator of consecutive `n`-sized chunks of list `l`.
        If `all` is `True`, returns **all** `n`-sized chunks in `l`
        by iterating over the starting point.
        """

        if all_:
            jarr = range(0, n - 1)
        else:
            jarr = [0]

        for j in jarr:
            for i in range(j, len(l), n):
                if i + 2 * n <= len(l):
                    yield l[i:i + n]
                else:
                    if not all_:
                        yield l[i:]
                    break

    # if remove_outliers:
    #     # Remove 5-sigma outliers from data
    #     # smoothed on a 1 day timescale
    #     if len(y) >= 50:
    #         ys = y - Smooth(y, 50)
    #     else:
    #         ys = y
    #     M = np.nanmedian(ys)
    #     MAD = 1.4826 * np.nanmedian(np.abs(ys - M))
    #     out = []
    #     for i, _ in enumerate(y):
    #         if (ys[i] > M + 5 * MAD) or (ys[i] < M - 5 * MAD):
    #             out.append(i)
    #     out = np.array(out, dtype=int)
    #     y = np.delete(y, out)

    running_cdpp = [np.std(yi) / np.sqrt(win) for yi in Chunks(y, win, all_=True)]

    return np.nanmedian(running_cdpp)


def calc_cdpp2(t, f, win):
    def block_mean(t, f, win):
        win_min = win / 2 + 1
        n = len(t)
        dt = np.median(t[1:] - t[:-1])
        t_blocks = []
        f_blocks = []
        i = 0
        while t[i] < t[-1]:
            j = np.copy(i)
            while (t[j] - t[i]) < (win * dt):
                j += 1
                if j >= n: break
            if j >= (i + win_min):
                t_blocks.append(t[i:j].mean())
                f_blocks.append(f[i:j].mean())
            i = np.copy(j)
            if i >= n: break
        t_blocks = np.array(t_blocks)
        f_blocks = np.array(f_blocks)
        return t_blocks, f_blocks

    # compute bin-averaged fluxes
    t_b, f_b = block_mean(t, f, win=win)
    cdpp = f_b.std()

    return cdpp


def compute_cdpp(time, flux, window, cadence=1626.0/60./60., robust=False):
    """
    Compute the CDPP in a given time window.
    :param time:
        The timestamps measured in days.
    :param flux:
        The time series. This should either be the raw data or normalized to
        unit mean (not relative flux with zero mean).
    :param window:
        The window in hours.
    :param cadence: (optional)
        The cadence of the observations measured in hours.
    :param robust: (optional)
        Use medians instead of means.
    :returns cdpp:
        The computed CDPP in ppm.
    """
    # Mask missing data and fail if no acceptable points exist.
    m = np.isfinite(time) * np.isfinite(flux)
    if not np.sum(m):
        return np.inf
    t, f = time[m], flux[m]

    # Compute the running relative std deviation.
    std = np.empty(len(t))
    hwindow = 0.5 * window
    for i, t0 in enumerate(t):
        m = np.abs(t - t0) < hwindow
        if np.sum(m) <= 0:
            std[i] = np.inf
        if robust:
            mu = np.median(f[m])
            std[i] = np.sqrt(np.median((f[m] - mu) ** 2)) / mu
        else:
            std[i] = np.std(f[m]) / np.mean(f[m])

    # Normalize by the window size.
    return np.median(std)   # / np.sqrt(window / cadence)


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    st_r, st_m = 0.456, 0.497
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


# run = "save_K2SC_mask_16_300_2000_1000"; lc_file = "LC_K2SC_mask.dat"
# run = "save_K2SC_16_250_2000_1000"; lc_file = "LC_K2SC.dat"
run = "save_K2SFF_16_300_2000_1000"; lc_file = "LC_K2SFF.dat"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)

rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
sas = keplerslaw(pers)

# t, f, e = np.loadtxt(lc_file, unpack=True, delimiter=",")
# t, f, e = data.LCtime, data.LC, data.LCerror

d = fits.open("hlsp_k2sff_k2_lightcurve_247887989-c13_kepler_v1_llc.fits")
dat = d[1].data    # T, FRAW, FCOR, ARCLENGTH, MOVING, CADENCENO
t = dat["T"]
f = dat["FCOR"]
t_u = np.linspace(t[0], t[-1], int((t[-1]-t[0])/0.020432106))   # uniform spacing
mf = medfilt(f, 25)     # median filter
cv = interpolate.interp1d(t, mf)(t_u)   # interpolate trend to full LC
gc = pyasl.broadGaussFast(t_u, cv, 0.05, edgeHandling="firstlast")  # gaussian convolve to smooth
gc = interpolate.interp1d(t_u, gc)(t)   # interpolate back to only data points from K2SFF

# plt.plot(t, f, ".")
# plt.plot(t, gc)
# _ = [plt.axvline(t0s[3]+i*pers[3], c="k") for i in range(3)]
# plt.show()

f = f - gc + 1.    # correct LC

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

    models.append(np.asarray(m))
    pars.append(params)


models = np.asarray(models)
m_tot = np.sum(models[:3], axis=0) - 2.     # total transit model for system

f01 = f - m_tot + 1.

res = f01 - models[3]   # residuals
cut = sigma_clip(f01, sigma_lower=8., sigma_upper=3.).mask
mask = cut == 0
t, f01, res = t[mask], f01[mask], res[mask]

# with open("01_lc.dat", "w") as dat:
#     for i in range(len(t)):
#         dat.write("{},{}\n".format(t[i], f01[i]))

# plt.plot(t, f01, ".")
# plt.plot(t, models[3])
# plt.show()

# plt.plot(t, f, ".")
# plt.plot(t, m_tot)
# plt.show()

p01, fp01 = phase_fold(t, f01, pers[3], t0s[3])
# _, fm01 = phase_fold(t, models[3], pers[3], t0s[3])
tss = np.linspace(t[0], t[-1], t.size*100)
bat = batman.TransitModel(pars[3], tss, supersample_factor=15, exp_time=29.4/60./24.0)
pm01, fm01 = phase_fold(tss, bat.light_curve(pars[3]), pers[3], t0s[3])

# std = np.std(f01)
std = np.sqrt(sum(res**2.) / float(t.size))
f_u = fp01 + std
f_l = fp01 - std
depth = 1. - min(fm01)

# cdpp = calc_cdpp2(t, f01, win=4)
# cdpp = compute_cdpp(t, f01, 2, robust=True)

print "Std =", int(std * 1e6), "ppm."
# print "CDPP =", cdpp
# print cdpp / std
print "Transit SNR = {:.2f}".format(depth / std)


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i in range(3):
    i1, i2 = i*t.size/3, (i+1)*t.size/3
    foldt = ((t[i1:i2] - t0s[3] + pers[3]/2.) / pers[3]) % 1
    foldf = f01[i1:i2]
    foldt, foldf = zip(*sorted(zip(foldt, foldf)))

    axes[i].errorbar(foldt, foldf, std, lw=0.8, marker=".", ms=10, elinewidth=1.5)    # capsize=4, capthick=1.5)
    for j in [-1., 0., 1.]:
        axes[i].axvline(0.5 + j*1./24./pers[3], c="k", ls="--", lw=0.8, alpha=0.7)
    # axes[i].set_xlim(0., 1.)
    # axes[i].set_xlim(0.495, 0.505)
    axes[i].plot(np.asarray(pm01), fm01, lw=5, alpha=0.4)
    axes[i].set_xlim(0.49, 0.51)
    axes[i].set_ylim(1.-8.*std, 1.+6*std)
plt.show()

plt.figure(figsize=(15, 4))
plt.errorbar(p01, fp01, std, ls="", marker=".", alpha=0.6)
# plt.fill_between(p01, f_u, f_l, edgecolor="k", facecolor="k", interpolate=True, alpha=0.2)
plt.plot(pm01, fm01, lw=2, c="g")
plt.axhline(1., lw=2, ls="--", c="g")
plt.axhline(min(fm01), lw=2, ls="--", c="g")
plt.xlim(0., 1.)
plt.show()

# binsize = 8
# p01 = np.asarray(p01)
# pbin = p01[:-1].reshape(t.size/binsize, binsize).mean(axis=1)
# rbin = res[:-1].reshape(t.size/binsize, binsize).mean(axis=1)
# plt.figure(figsize=(12, 4))
# plt.errorbar(pbin, rbin, std/np.sqrt(float(binsize)), ls="", marker=".", alpha=0.6)
# plt.axhline(0.)
# plt.xlim(0.48, 0.52)
# plt.show()


# bls = BLS(t, f01, np.ones(t.size, float)*std, period_range=(15, 60), q_range=(0.002, 0.1), nf=10000, nbin=t.size)
# res = bls()  # do BLS search
# p_ar, p_pow, t0, rprs, bls_depth = bls.period, res.p, bls.tc, np.sqrt(res.depth), res.depth
# trend = medfilt(p_pow, 201)
# rmed_pow = p_pow - trend  # subtract running median
# p_sde = rmed_pow / rmed_pow.std()  # SDE
# sde_trans = np.nanmax(p_sde)  # highest SDE
# bper = p_ar[np.argmax(p_sde)]  # corresponding period

# fig1, ax = plt.subplots(1, 1, figsize=(14, 6))
# ax.plot(p_ar, p_pow)
# ax.plot(p_ar, trend)
# plt.show()

# fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
# ax.plot(p_ar, p_sde, lw=2)  # BLS spectrum
# ax.plot(bper, sde_trans, ls='', marker='*', ms=15)  # highest SDE
# ax.set_xlim(min(p_ar), max(p_ar))
# ax.set_ylim(0.)
# plt.show()


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# tpf = lk.KeplerTargetPixelFile.from_archive(247887989)
# lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
#
# lc = lk.KeplerLightCurveFile.from_archive(247887989).PDCSAP_FLUX
# lc = lc.flatten(window_length=401)
# lc = lc.fold(period=11.0243)
# lc = lc.bin(binsize=10)
# lc.plot(ls="", marker=".")
# plt.show()
