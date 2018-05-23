from __future__ import print_function
import dill
import matplotlib.pyplot as plt
import numpy as np
import batman
from exoplanet import phase_fold
import pandas
from scipy import interpolate
from astropy.io import ascii
import sys
from exoplanet import phase_fold
import pandas
import my_constants as myc


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    st_r, st_m = 0.46, 0.50
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


run = "save_K2SC_mask_16_250_2000_1000"; lc_file = "LC_K2SC_mask.dat"
# run = "save_K2SC_16_250_2000_1000"; lc_file = "LC_K2SC"
# run = "save_K2SFF_16_250_2000_1000"; lc_file = "LC_K2SFF"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)

rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
sas = keplerslaw(pers)


lines = np.loadtxt(run + "/fit_vals.dat", delimiter="\n", dtype=tuple)  # read MCMC output
all_vals = []
for i in range(16):
    line = lines[i].strip("(").strip(")")
    vals = np.asarray([round(float(v), 4) for v in line.split(", ")])
    all_vals += list(vals)
props = []
for prop in ["Rp/Rs", "Inclination", "Epoch", "Period"]:
    props += [prop, prop+"_u_err", prop+"_l_err"]
df = pandas.DataFrame(columns=["Property", "b", "c", "d", "01"])
df["Property"] = props
for j in range(4):
    col = ["d", "c", "b", "01"][j]
    pl_vals = all_vals[j*12:j*12+12]
    df[col] = pl_vals

df.loc[len(df)] = ["Rp"] + list(np.round(rps * 0.46 * myc.RS / myc.RE, 2))
df.loc[len(df)] = ["a"] + list(np.round(sas, 1))

df.to_csv(run + "/fit_vals.csv", index=False)


# data
t, f, e = np.loadtxt(lc_file, unpack=True, delimiter=",")
# t, f, e = data.LCtime, data.LC, data.LCerror

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
    p2, mp = phase_fold(t, m, per, t0+per/4.)   # phase-folded model

    models.append(np.asarray(m))
    pars.append(params)


models = np.asarray(models)
m_tot = np.sum(models, axis=0) - 3.     # total transit model for system

# with open("bat_pars.pkl", "wb") as pf:
#     dill.dump(pars, pf)
# sys.exit()

for i in range(4):
    pl = ["b", "c", "d", "01"][i]
    others = [0, 1, 2, 3]
    others.remove(i)
    m_others = np.ones(m_tot.size, dtype=float)
    for j in others:
        m_others += (models[j] - 1.)

    lc = f - m_others + 1.

    # plt.plot(t, lc, ".")
    # plt.plot(t, models[i])
    # plt.show()

    t_ss = np.linspace(t.min(), t.max(), t.size * 100)  # supersample
    bat = batman.TransitModel(pars[i], t_ss, supersample_factor=15, exp_time=29.4/60./24.0)
    m_ss = bat.light_curve(pars[i])
    period = pers[i]
    t0 = t0s[i]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3, 1]})

    # plot full LC and single model
    ax1.plot(t, lc, 'o', ms=3, alpha=0.7, zorder=1)
    ax1.plot(t_ss, m_ss, lw=2, alpha=0.8, zorder=2)
    ax1.set_xlim([min(t), max(t)])

    # plot phase-folded LC and ss-single model
    phase, lcfold = phase_fold(t, lc, period, t0)
    phase = np.asarray(phase)
    phase = (phase - 0.5) * period * 24.  # hours from transit centre
    ax2.plot(phase, lcfold, 'o', ms=3, alpha=0.7, zorder=1)
    mphase, mfold = phase_fold(t_ss, m_ss, period, t0)
    mphase = np.asarray(mphase)
    mphase = (mphase - 0.5) * period * 24.
    ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2)
    ax2.set_xlim(-3, 3)
    # ax2.set_ylim(0.9975, 1.001)
    ax2.tick_params(axis='x', labelbottom=False)

    # plot phase-folded residuals
    _, mfold_lct = phase_fold(t, models[i], period, t0)
    resid = np.asarray(lcfold) - np.asarray(mfold_lct)
    ax3.plot(phase, resid, 'o', alpha=0.7, ms=3, zorder=1)
    ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=2)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim([-np.std(resid) * 5., np.std(resid) * 5.])

    ax2.get_shared_x_axes().join(ax2, ax3)
    plt.tight_layout()
    plt.savefig(run + "/new_fit_{}.pdf".format(pl), format="pdf")
    # plt.show()
    plt.close()


# diff = f - m_tot    # residuals
#
# devp = 4*np.std(diff)       # std above
# devm = -2.8*np.std(diff)      # std below
# msk = (diff <= devp) & (diff >= devm)   # good points
# msk2 = np.argwhere((diff > devp) | (diff < devm))[:,0]  # indices of bad points
#
# print(len(msk) - sum(msk), "\n")
# print(list(msk2))
#
# plt.plot(t[msk], diff[msk], ".")
# plt.plot(t[np.invert(msk)], diff[np.invert(msk)], "x")
# plt.axhline(0., color="k")
# plt.axhline(devp, color="k", ls="--")
# plt.axhline(devm, color="k", ls="--")
# plt.show()


# def bls_search(time, flux, err, per_range=(1., 80.), q_range=(0.001, 0.115), nf=10000, nbin=900, plot=False):
#     from pybls import BLS
#     from scipy.signal import medfilt as mf
#     bls = BLS(time, flux, err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
#     res = bls()  # do BLS search
#
#     sde_trans, p_ar, p_pow, bper, t0, rprs, depth, p_sde = \
#         res.bsde, bls.period, res.p, res.bper, bls.tc, np.sqrt(res.depth), res.depth, res.sde
#
#     test_pow = p_pow - mf(p_pow, 201)  # running median
#     p_sde = test_pow / test_pow.std()  # SDE
#     sde_trans = np.nanmax(p_sde)
#     bper = p_ar[(np.abs(p_sde - sde_trans)).argmin()]
#
#     pr = (res.in2 - res.in1) / 24. / bper  # limit plot to 2 durations in phase
#
#     if plot:
#         plt.figure(figsize=(15, 8))
#         plt.plot(p_ar, p_sde, lw=2)
#         plt.xlim([min(p_ar), max(p_ar)])
#         plt.ylim([0., int(sde_trans) + 1])
#         plt.savefig("BLS.pdf")
#         plt.close("all")
#
#         p, f = phase_fold(time, flux, bper, t0)
#         fig, ax = plt.subplots(1, figsize=(15, 8))
#         ax.plot(p, f, ".")
#         ax.set_xlim(0.5-pr, 0.5+pr)
#         plt.savefig("BLS_phase_fold.pdf")
#         plt.close("all")
#
#     return sde_trans, bper, t0, rprs, pr
#
#
# sde_trans, bper, t0, rprs, pr = bls_search(t, f-m_tot+1., np.ones(t.size)*1e-4, plot=True)
# print(">>> SDE = {}, Per = {} days, Epoch = {}".
#       format(round(sde_trans, 1), round(bper, 2), round(t0, 3)))
