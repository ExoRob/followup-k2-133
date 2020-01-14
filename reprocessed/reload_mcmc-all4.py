# from __future__ import print_function
import dill
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import numpy as np
import batman
from my_exoplanet import phase_fold
import pandas
from scipy import interpolate
from astropy.io import ascii
import sys
import pandas
from tqdm import tqdm
import my_constants as myc
import corner
sys.path.append("/Users/rwells/Documents/LDC3/python")
import LDC3
from copy import deepcopy
import seaborn as sns
sns.set()
sns.set_style("ticks")
sns.set_color_codes()

st_r, st_m = 0.455, 0.461
st_r_e, st_m_e = 0.022, 0.011

confs = [50.-68.27/2., 50., 50.+68.27/2.]
# confs = [50.-95.45/2., 50., 50.+95.45/2.]
# confs = [50.-99.73/2., 50., 50.+99.73/2.]


def single_model(pars, t, pl):
    # set self.params for batman model
    params = batman.TransitParams()
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"

    c2, c3, c4 = LDC3.forward(pars[-3:])
    params.u = [0., c2, c3, c4]

    params.rp = pars[0 + pl * 4]
    params.inc = pars[1 + pl * 4]
    params.t0 = pars[2 + pl * 4]
    params.per = pars[3 + pl * 4]
    params.a = keplerslaw(params.per)
    batman_model = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    light_curve = batman_model.light_curve(params)

    return light_curve


def get_transit_curve(pars, t):
    """ return the transit curve at times t """
    light_curve = np.ones(t.size, dtype=float)

    # set self.params for batman model
    params = batman.TransitParams()
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"

    c2, c3, c4 = LDC3.forward(pars[-3:])
    params.u = [0., c2, c3, c4]

    for pl in range(4):
        params.rp = pars[0+pl*4]
        params.inc = pars[1+pl*4]
        params.t0 = pars[2+pl*4]
        params.per = pars[3+pl*4]
        params.a = keplerslaw(params.per)
        batman_model = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        light_curve += batman_model.light_curve(params) - 1.

    return light_curve


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def keplerslaw_stellar(kper, stellar_m, stellar_r):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = stellar_m * 1.989e30  # kg
    Rstar = stellar_r * 695700000.  # m
    G = 6.67408e-11  # m3 kg-1 s-2

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


run = "mcmc/save_broad_19_300_4000_2600"
# run = "mcmc/save_broad_19_250_1500_800"

pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)


# make LDC corner plot
ldc_samples = samples.T[-3:]

cs = np.array([np.asarray(LDC3.forward(alphas)) for alphas in ldc_samples.T])

for i in range(3):
    cl, c, cu = np.percentile(cs.T[i], confs)
    print c, c-cl, cu-c
# sys.exit()

allsamples = np.array([samples.T[i] for i in range(12, 16)] + [cs.T[i] for i in range(3)])

fig = corner.corner(allsamples.T, labels=["\n$R_p/R_s$", "\n$i$", "\n$T_0$", "\n$P$", "\n$C_2$", "\n$C_3$", "\n$C_4$"],
                    bins=30, use_math_text=True, plot_contours=True,
                    label_kwargs={"fontsize": 35}, hist2d_kwargs={"bins": 30})
for ax in fig.get_axes():
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)
fig.subplots_adjust(bottom=0.1, left=0.07)
plt.savefig(run + "/new_samples.pdf", format="pdf")
sys.exit()

#
# params = batman.TransitParams()
# params.t0 = 0.
# params.per = 10.
# params.rp = 0.1
# params.a = 3.
# params.inc = 90.
# params.ecc = 0.
# params.w = 90.
# params.limb_dark = "nonlinear"
# t = np.linspace(-1, 1, 1000)
# for c in tqdm(cs[:1000]):
#     params.u = [0., c[0], c[1], c[2]]
#
#     bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.0)
#     m = bat.light_curve(params)
#
#     plt.plot(t, m, c="0.5", alpha=0.3)
# plt.show()
#
# corner.corner(allsamples.T, labels=["a-h", "a-r", "a-t", "c2", "c3", "c4"], use_math_text=True, plot_contours=True)
#               # label_kwargs={"fontsize": 20}, hist2d_kwargs={"bins": 50})
# # plt.savefig("new_samples.pdf", format="pdf")
# # plt.close("all")
# plt.show()
# sys.exit()

# print planet.stellar_radius, planet.stellar_mass

# a1 = np.transpose(samples)[3]
# a2 = np.random.normal(0.1, 0.001, samples.shape[0])
#
# print a1 * 100. * 0.1
#
# # wrong_dist = samples.T[0] * np.random.normal(st_r, st_r_e, samples.shape[0]) * 100.
# wrong_dist = a1 * a2 * 100.
# # print type(samples)
# # print type(samples.T[0]), samples.T[0].mean()
# # print type(np.random.normal(st_r, st_r_e, samples.shape[0])), np.random.normal(st_r, st_r_e, samples.shape[0]).mean()
# print a1.mean(), "X", a2.mean(), "X", 100, "=", wrong_dist.mean()
# sys.exit()

rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
sas = keplerslaw(pers)
cs = np.array([planet.alpha_h, planet.alpha_r, planet.alpha_t])

# lines = np.loadtxt(run + "/fit_vals.dat", delimiter="\n", dtype=tuple)  # read MCMC output
# all_vals = []
# for i in range(16):
#     line = lines[i].strip("(").strip(")")
#     vals = np.asarray([round(float(v), 4) for v in line.split(", ")])
#     all_vals += list(vals)
# props = []
# for prop in ["Rp/Rs", "Inclination", "Epoch", "Period"]:
#     props += [prop, prop+"_u_err", prop+"_l_err"]
# df = pandas.DataFrame(columns=["Property", "b", "c", "d", "01"])
# df["Property"] = props
# for j in range(4):
#     col = ["d", "c", "b", "01"][j]
#     pl_vals = all_vals[j*12:j*12+12]
#     df[col] = pl_vals
#
# df.loc[len(df)] = ["Rp"] + list(np.round(rps * 0.456 * myc.RS / myc.RE, 2))
# df.loc[len(df)] = ["a"] + list(np.round(sas, 1))
#
# df.to_csv(run + "/fit_vals.csv", index=False)

st_r_sample = np.random.normal(st_r, st_r_e, samples.shape[0])
st_m_sample = np.random.normal(st_m, st_m_e, samples.shape[0])

pcs = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, confs, axis=0)))

for pl in range(4):
    rp, rp_u, rp_l = pcs[0+pl*4]
    inc, inc_u, inc_l = pcs[1+pl*4]
    t0, t0_u, t0_l = pcs[2+pl*4]
    per, per_u, per_l = pcs[3+pl*4]
    # a = keplerslaw(per)

    # rprs x Rs
    r_l, r, r_u = np.percentile(samples.T[0+pl*4] * st_r_sample * myc.RS / myc.RE, confs)

    # plt.hist(samples.T[0+pl*4] * np.random.normal(st_r, st_r_e, samples.shape[0]) * myc.RS / myc.RE, 200)
    # _ = [plt.axvline(v, color="k", ls="--") for v in [r_l, r, r_u]]
    # plt.show()

    # a/Rs
    a_rs_l, a_rs, a_rs_u = np.percentile(keplerslaw_stellar(samples.T[3+pl*4], st_m_sample, st_r_sample), confs)

    # a/Rs * Rs
    a_au_l, a_au, a_au_u = np.percentile(keplerslaw_stellar(samples.T[3+pl*4], st_m_sample, st_r_sample) *
                                         st_r_sample * myc.RS / myc.AU, confs)

    # a/Rs * arccos i
    b_l, b, b_u = np.percentile(calc_b(keplerslaw_stellar(samples.T[3+pl*4], st_m_sample, st_r_sample),
                                       samples.T[1+pl*4]), confs)

    s = """
    $T_{{0}}$ (BJD) & ${:.5f}_{{-{:.5f}}}^{{+{:.5f}}}$ \\\\ [0.1cm]
    Period (days) & ${:.5f}_{{-{:.5f}}}^{{+{:.5f}}}$ \\\\ [0.1cm]
    $R_{{p}}/R_{{s}}$ & ${:.4f}_{{-{:.4f}}}^{{+{:.4f}}}$ \\\\ [0.1cm]
    Inclination (deg.) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
     & \\\\
    Derived properties: & \\\\ [0.1cm]
    Radius ($R_{{\earth}}$) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
    Semi-major axis (sr) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
    Semi-major axis (au) & ${:.5f}_{{-{:.5f}}}^{{+{:.5f}}}$ \\\\ [0.1cm]
    Impact parameter & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]"""\
        .format(t0, t0_l, t0_u,
                per, per_l, per_u,
                rp, rp_l, rp_u,
                inc, inc_l, inc_u,
                r, r-r_l, r_u-r,
                a_rs, a_rs-a_rs_l,
                a_rs_u-a_rs, a_au, a_au-a_au_l,
                a_au_u-a_au, b, b-b_l, b_u-b)

    print s, "\n"

print [906.5418288, 987.34108546, 1073.24383493], [1.3844646, 1.44826481, 1.53758511]

sys.exit()

# ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---

# find_bestfit = True
find_bestfit = False
# find_depth_dur = True
find_depth_dur = False

inds = np.random.randint(0, samples.shape[0], 25000)
# inds = np.arange(samples.shape[0])

nt = 20000


# find best-fit values
if find_bestfit:
    chisq = 1e10

    for i, ind in enumerate(tqdm(inds, desc="Finding best-fit values")):
        m_full = get_transit_curve(samples[ind], data.LCtime)

        new_chisq = sum((data.LC - m_full)**2.)
        if new_chisq < chisq:
            chisq = deepcopy(new_chisq)
            bestfit_model = np.copy(m_full)
            bestfit_pars = deepcopy(samples[ind])
            # print chisq

    with open(run + "/bestfit.pkl", "wb") as pf:
        dill.dump([bestfit_pars, bestfit_model], pf)

else:
    with open(run + "/bestfit.pkl", "rb") as pf:
        bestfit_pars, bestfit_model = dill.load(pf)


rp, inc, t0, per = bestfit_pars[12:16]
t_ss = np.linspace(t0 - 0.1*per, t0 + 0.1*per, nt)

# find duration and depth
if find_depth_dur:
    m_rs = np.ones((inds.size, t_ss.size), float)      # the model vs t_ss
    m_rs_t = np.ones((inds.size, data.LCtime.size), float)  # the model vs t
    p_rs = []

    params_rs = batman.TransitParams()      # batman parameters
    params_rs.ecc = 0.
    params_rs.w = 90.
    params_rs.limb_dark = "nonlinear"

    depths = np.zeros(inds.size, float)
    durations = np.zeros(inds.size, float)
    for i, ind in enumerate(tqdm(inds, desc="Finding duration and depth")):
        params_rs.rp = samples[ind][0+3*4]
        params_rs.inc = samples[ind][1+3*4]
        params_rs.t0 = samples[ind][2+3*4]
        params_rs.per = samples[ind][3+3*4]
        params_rs.a = keplerslaw(params_rs.per)

        c2, c3, c4 = LDC3.forward(samples[ind][-3:])
        params_rs.u = [0., c2, c3, c4]

        m_rs[i] += batman.TransitModel(params_rs, np.linspace(params_rs.t0 - 0.1*params_rs.per,
                                                              params_rs.t0 + 0.1*params_rs.per, nt),
                                       supersample_factor=15, exp_time=29.4 / 60. / 24.0).light_curve(params_rs) - 1.
        m_rs_t[i] += batman.TransitModel(params_rs, data.LCtime,
                                         supersample_factor=15, exp_time=29.4 / 60. / 24.0).light_curve(params_rs) - 1.
        p_rs.append(deepcopy(params_rs))

        depths[i] = (1. - batman.TransitModel(params_rs, np.array([params_rs.t0])).light_curve(params_rs)[0]) * 1e6

        inds_in_transit = np.argwhere(batman.TransitModel(params_rs, np.linspace(params_rs.t0 - 0.1*params_rs.per,
                                                                                 params_rs.t0 + 0.1*params_rs.per, nt)).
                                      light_curve(params_rs) != 1.)
        durations[i] = float(inds_in_transit[-1] - inds_in_transit[0]) * (t_ss[-1] - t_ss[0]) / nt * 24.

        # plt.plot(m_rs[i], c="0.4", alpha=0.4)
    # plt.show()

    print np.percentile(depths, confs)
    print np.percentile(durations, confs)

    with open(run + "/depth_dur.pkl", "wb") as pf:
        dill.dump([p_rs, depths, durations], pf)

else:
    with open(run + "/depth_dur.pkl", "rb") as pf:
        p_rs, depths, durations = dill.load(pf)

    m_rs = np.ones((inds.size, t_ss.size), float)  # the model vs t_ss
    m_rs_t = np.ones((inds.size, data.LCtime.size), float)  # the model vs t
    for i in tqdm(range(inds.size), desc="Retrieving model samples"):
        params_rs = p_rs[i]
        m_rs[i] += batman.TransitModel(p_rs[i], np.linspace(params_rs.t0 - 0.1 * params_rs.per,
                                                            params_rs.t0 + 0.1 * params_rs.per, nt),
                                       supersample_factor=15, exp_time=29.4 / 60. / 24.0).light_curve(params_rs) - 1.

        m_rs_t[i] += batman.TransitModel(p_rs[i], data.LCtime,
                                         supersample_factor=15, exp_time=29.4 / 60. / 24.0).light_curve(params_rs) - 1.


# ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---

# data
t, f, e = data.LCtime, data.LC, data.LCerror

models = [single_model(bestfit_pars, data.LCtime, pl) for pl in range(4)]

# m_tot = np.sum(models, axis=0) - 3.     # total transit model for system
# with open("pars.pkl", "wb") as pf:
#     dill.dump(pars, pf)

# make plot for planet e
phplot = np.linspace(-0.1, 0.1, nt)
# m_ss = single_model(bestfit_pars, t_ss, 3)

fig, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [8, 2]})

# plot phase-folded LC and ss-single model
phase, lcfold = phase_fold(t, f-np.sum(models[:-1], axis=0)+3., per, t0)
phase = np.asarray(phase)
phase = (phase - 0.5) * per * 24.  # hours from transit centre
ax2.errorbar(phase, lcfold, e, lw=0., marker=".", ms=10, elinewidth=1.5, color="k", zorder=2)
ax2.plot(phase, lcfold, ".", ms=4, lw=0.9, ls="", zorder=3, color="b")

# mphase, mfold = phase_fold(t_ss, m_ss, per, t0)
# mphase = np.asarray(mphase)
# mphase = (mphase - 0.5) * per * 24.
# ax2.plot(mphase, mfold, lw=1.5, zorder=4, color="g", ls="--")

for i, pc in enumerate([[50.-68.27/2., 50., 50.+68.27/2.],       # 1-sigma
                        [50.-95.45/2., 50., 50.+95.45/2.]]):     # 2-sigma
    ml, mm, mh = np.percentile(m_rs, pc, axis=0)    # low, mid, high
    resid_l, resid_h = mm - ml, mm - mh             # model residuals

    ax2.plot(phplot*per*24., mm, lw=1.5, zorder=3, color="k")

    ax2.fill_between(phplot*per*24., ml, mh, ml < mh, color="grey", alpha=0.5 if (i == 0) else 0.3)

    ax3.fill_between(phplot*per*24., resid_l, resid_h, resid_h < resid_l, color="grey",
                     alpha=0.5 if (i == 0) else 0.3)

# plot phase-folded residuals
mid_t_model = np.median(m_rs_t, axis=0)
resid = np.asarray(lcfold) - np.asarray(phase_fold(t, mid_t_model, per, t0)[1])
# ax3.plot(phase, resid, 'k.', ms=10, zorder=1)
ax3.errorbar(phase, resid, e, lw=0., marker=".", ms=10, elinewidth=1.5, color="k", zorder=2)
ax3.plot(phase, resid, 'b.', ms=4, zorder=3)
ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=0)

ax3.set_ylim([-np.std(resid) * 4., np.std(resid) * 4.])

ax2.set_xlim(-3.05, 3.05)
ax2.set_ylim(0.9985, 1.0005)
ax3.set_ylim(-0.0005, 0.0005)
ax2.yaxis.set_major_locator(tic.MultipleLocator(base=0.0005))
ax2.yaxis.set_minor_locator(tic.MultipleLocator(base=0.0001))
ax3.yaxis.set_major_locator(tic.MultipleLocator(base=0.0005))
ax3.yaxis.set_minor_locator(tic.MultipleLocator(base=0.0001))
ax2.xaxis.set_major_locator(tic.MultipleLocator(base=1.))
ax2.xaxis.set_minor_locator(tic.MultipleLocator(base=0.2))

ax2.tick_params(axis='x', labelbottom=False)
ax2.set_ylabel("Normalised flux", fontsize=15)
ax3.set_xlabel("Time from transit centre (hours)", fontsize=15)
ax3.set_ylabel("Residuals", fontsize=15)
# fig.text(0.01, 0.5, 'Normalised flux', ha='center', va='center', rotation='vertical', fontsize=15)
plt.tight_layout()
plt.savefig(run + "/new_fit.pdf", format="pdf")
# plt.show()
plt.close()

# ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---

# for i in range(4):
#     pl = ["b", "c", "d", "01"][i]
#     others = [0, 1, 2, 3]
#     others.remove(i)
#     m_others = np.ones(m_tot.size, dtype=float)
#     for j in others:
#         m_others += (models[j] - 1.)
#
#     lc = f - m_others + 1.
#
#     # plt.plot(t, lc, ".")
#     # plt.plot(t, models[i])
#     # plt.show()
#
#     t_ss = np.linspace(t.min(), t.max(), t.size * 100)  # supersample
#     bat = batman.TransitModel(pars[i], t_ss, supersample_factor=15, exp_time=29.4/60./24.0)
#     m_ss = bat.light_curve(pars[i])
#     period = pers[i]
#     t0 = t0s[i]
#
#     fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3, 1]})
#
#     # plot full LC and single model
#     ax1.plot(t, lc, 'o', ms=3, alpha=0.7, zorder=1)
#     ax1.plot(t_ss, m_ss, lw=2, alpha=0.8, zorder=2)
#     ax1.set_xlim([min(t), max(t)])
#
#     # plot phase-folded LC and ss-single model
#     phase, lcfold = phase_fold(t, lc, period, t0)
#     phase = np.asarray(phase)
#     phase = (phase - 0.5) * period * 24.  # hours from transit centre
#     ax2.plot(phase, lcfold, 'o', ms=3, alpha=0.7, zorder=1)
#     mphase, mfold = phase_fold(t_ss, m_ss, period, t0)
#     mphase = np.asarray(mphase)
#     mphase = (mphase - 0.5) * period * 24.
#     ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2)
#     ax2.set_xlim(-3, 3)
#     # ax2.set_ylim(0.9975, 1.001)
#     ax2.tick_params(axis='x', labelbottom=False)
#
#     # plot phase-folded residuals
#     _, mfold_lct = phase_fold(t, models[i], period, t0)
#     resid = np.asarray(lcfold) - np.asarray(mfold_lct)
#     ax3.plot(phase, resid, 'o', alpha=0.7, ms=3, zorder=1)
#     ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=2)
#     ax3.set_xlim(-3, 3)
#     ax3.set_ylim([-np.std(resid) * 5., np.std(resid) * 5.])
#
#     ax2.get_shared_x_axes().join(ax2, ax3)
#     plt.tight_layout()
#     plt.savefig(run + "/new_fit_{}.pdf".format(pl), format="pdf")
#     # plt.show()
#     plt.close()


# diff = f - m_tot    # residuals
#
# devp = 4.*np.std(diff)       # std above
# devm = -4.*np.std(diff)      # std below
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
