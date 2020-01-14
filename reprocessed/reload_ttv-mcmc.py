import dill
import matplotlib.pyplot as plt
from matplotlib import ticker as tic
import numpy as np
import batman
from my_exoplanet import phase_fold
import pandas
from tqdm import tqdm
import my_constants as myc
import corner
import seaborn as sns
sns.set()
sns.set_style("ticks")
sns.set_color_codes()

st_r, st_m = 0.455, 0.461
st_r_e, st_m_e = 0.022, 0.011
G = 6.67408e-11  # m3 kg-1 s-2


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * 1.989e30  # kg
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def keplerslaw_stellar(kper, stellar_m, stellar_r):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = stellar_m * 1.989e30  # kg
    Rstar = stellar_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


pklfile = "ttv-mcmc.pkl"
with open(pklfile, "rb") as pklf:
    all_data, all_planet, all_samples = dill.load(pklf)

run = "save_1_16_150_1600_800"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    fitted_data, fitted_planet, fitted_samples = dill.load(pklf)


fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
for pl in range(4):
    t0 = np.median(fitted_samples.T[2 + pl * 4])
    per = np.median(fitted_samples.T[3 + pl * 4])
    # n_trans = int(np.floor((fitted_data.LCtime[-1] - t0) / per) + 1.)

    bad_transits = [[],
                    [14],
                    [15, 23],
                    []]
    ax = [2, 1, 0, 3][pl]

    x, y, yl, yu = [], [], [], []
    for i in range(len(all_data[pl])):
        if i not in bad_transits[pl]:
            samples = np.asarray(all_samples[pl][i]).flatten()

            pc = np.percentile(samples, [50.-68.27/2., 50, 50.+68.27/2.])

            y.append(pc[1] - t0 - i*per)
            yl.append(pc[1] - pc[0])
            yu.append(pc[2] - pc[1])
            x.append(t0+i*per)

            # axes[ax].text(x[-1], y[-1], i)

    y = np.asarray(y) * 24. * 60.
    yl = np.asarray(yl) * 24. * 60.
    yu = np.asarray(yu) * 24. * 60.

    print np.mean(np.asarray(yl)/2. + np.asarray(yu)/2.)

    axes[ax].errorbar(x, y, yerr=[yl, yu], fmt='--o', ms=8, elinewidth=1.6, label="b,c,d,e".split(",")[pl])

    axes[ax].axhline(0., color="0.4", lw=3, alpha=0.6)

    lim = max(np.abs(np.array([min(y - yl), max(y + yu)]))) + 1
    axes[ax].set_ylim(-lim, lim)

    axes[ax].text(2988, -lim/1.2, "d,c,b,e".split(",")[pl], fontsize=20, fontweight='bold')

    axes[ax].xaxis.set_major_locator(tic.MultipleLocator(base=10.))
    axes[ax].xaxis.set_minor_locator(tic.MultipleLocator(base=5.))
    axes[ax].yaxis.set_major_locator(tic.MultipleLocator(base=5.))
    axes[ax].yaxis.set_minor_locator(tic.MultipleLocator(base=1.))

# plt.legend()
axes[3].set_xlabel("Time (BKJD)", fontsize=18)
fig.text(0.0, 0.5, 'TTV (min)', ha='left', va='center', rotation='vertical', fontsize=18)
plt.tight_layout(0.)
fig.subplots_adjust(left=0.07)
plt.savefig("k2-133-ttv.pdf")
plt.show()

import sys; sys.exit()

# corner plot the MCMC samples
corner.corner(samples, labels=planet.labels, bins=50, use_math_text=True, plot_contours=True,   # fig=corner_fig,
              label_kwargs={"fontsize": 15}, hist2d_kwargs={"bins": 50})
plt.savefig("new_samples.pdf", format="pdf")
# plt.close("all")
plt.show()


t, f, e = data.LCtime, data.LC, data.LCerror

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

# model matching t
bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.0)
m = bat.light_curve(params)

nt = 20000      # number of phase points
ns = 5000      # number of samples of model
t_ss = np.linspace(t0 - 0.1*per, t0 + 0.1*per, nt)   # phase - super-sampled
m_rs = np.ones((ns, t_ss.size), float)      # the model vs t_ss
params_rs = batman.TransitParams()      # batman parameters
params_rs.t0 = t0
params_rs.per = per
params_rs.rp = rp
params_rs.a = a
params_rs.inc = inc
params_rs.ecc = 0.
params_rs.w = 90.
params_rs.u = [0.5079, 0.2239]
params_rs.limb_dark = "quadratic"
inds = np.random.randint(0, samples.shape[0], ns)
depths = np.zeros(ns, float)
durations = np.zeros(ns, float)
for i, ind in enumerate(tqdm(inds)):
    params_rs.rp = samples[ind][0]
    params_rs.inc = samples[ind][1]
    params_rs.t0 = samples[ind][2]
    params_rs.per = samples[ind][3]
    params_rs.a = keplerslaw(params_rs.per)

    bat_rs = batman.TransitModel(params_rs, np.linspace(params_rs.t0 - 0.1*params_rs.per,
                                                        params_rs.t0 + 0.1*params_rs.per, nt),
                                 supersample_factor=15, exp_time=29.4 / 60. / 24.0)
    m_rs[i] += bat_rs.light_curve(params_rs) - 1.

    # depths[i] = (1. - batman.TransitModel(params_rs, np.array([params_rs.t0]), supersample_factor=15,
    #                                       exp_time=29.4 / 60. / 24.0).light_curve(params_rs)[0]) * 1e6
    # inds_in_transit = np.argwhere(m_rs[i] != 1.)
    # durations[i] = float(inds_in_transit[-1] - inds_in_transit[0]) * (t_ss[-1] - t_ss[0]) / nt * 24.

# print np.percentile(depths, [5, 50, 95])
# print np.percentile(durations, [5, 50, 95])

phplot = np.linspace(-0.1, 0.1, nt)

bat = batman.TransitModel(params, t_ss, supersample_factor=15, exp_time=29.4/60./24.0)
m_ss = bat.light_curve(params)

fig, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [8, 2]})

# plot phase-folded LC and ss-single model
phase, lcfold = phase_fold(t, f, per, t0)
phase = np.asarray(phase)
phase = (phase - 0.5) * per * 24.  # hours from transit centre
ax2.errorbar(phase, lcfold, e, lw=0., marker=".", ms=10, elinewidth=1.5, color="k", zorder=2)
ax2.plot(phase, lcfold, ".", ms=4, lw=0.9, ls="", zorder=3, color="b")

# mphase, mfold = phase_fold(t_ss, m_ss, per, t0)
# mphase = np.asarray(mphase)
# mphase = (mphase - 0.5) * per * 24.
# ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2)

for i, pc in enumerate([[100.-68.27, 50., 68.27],       # 1-sigma
                        [100.-95.45, 50., 95.45]]):     # 2-sigma
    ml, mm, mh = np.percentile(m_rs, pc, axis=0)    # low, mid, high
    resid_l, resid_h = mm - ml, mm - mh             # model residuals

    ax2.plot(phplot*per*24., mm, color="k", lw=1.5)
    ax2.fill_between(phplot*per*24., ml, mh, ml < mh, color="grey", alpha=0.5 if (i == 0) else 0.3)

    ax3.fill_between(phplot*per*24., resid_l, resid_h, resid_h < resid_l, color="grey",
                     alpha=0.5 if (i == 0) else 0.3)

# plot phase-folded residuals
_, mfold_lct = phase_fold(t, m, per, t0)
resid = np.asarray(lcfold) - np.asarray(mfold_lct)
ax3.plot(phase, resid, 'k.', ms=10, zorder=1)
ax3.plot(phase, resid, 'b.', ms=4, zorder=1)
ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=0)

ax3.set_ylim([-np.std(resid) * 4., np.std(resid) * 4.])

ax2.set_xlim(-3.05, 3.05)
ax2.set_ylim(0.9985, 1.0005)
ax3.set_ylim(-0.0004, 0.0004)
ax2.yaxis.set_major_locator(tic.MultipleLocator(base=0.0005))
ax2.yaxis.set_minor_locator(tic.MultipleLocator(base=0.0001))
ax3.yaxis.set_major_locator(tic.MultipleLocator(base=0.0004))
ax3.yaxis.set_minor_locator(tic.MultipleLocator(base=0.0001))
ax2.xaxis.set_major_locator(tic.MultipleLocator(base=1.))
ax2.xaxis.set_minor_locator(tic.MultipleLocator(base=0.2))

ax2.tick_params(axis='x', labelbottom=False)
ax2.set_ylabel("Normalised flux", fontsize=15)
ax3.set_xlabel("Time from transit centre (hours)", fontsize=15)
ax3.set_ylabel("Residuals", fontsize=15)
# fig.text(0.01, 0.5, 'Normalised flux', ha='center', va='center', rotation='vertical', fontsize=15)
plt.tight_layout()
# plt.savefig("new_fit.pdf", format="pdf")
plt.show()
# plt.close()
