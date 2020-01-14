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

st_r, st_m = 0.4547, 0.4608
st_r_e, st_m_e = 0.0104, 0.0051

G = 6.67408e-11  # m3 kg-1 s-2

confs = [50.-68.2689/2., 50., 50.+68.2689/2.]
# confs = [5., 50., 95.]


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


pklfile = "mcmc/mcmc-short.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)

st_r_sample = np.random.normal(st_r, st_r_e, samples.shape[0])
st_m_sample = np.random.normal(st_m, st_m_e, samples.shape[0])

pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, confs, axis=0)))

# load median values from MCMC fit
rp, rp_u, rp_l = pcs[0]
inc, inc_u, inc_l = pcs[1]
t0, t0_u, t0_l = pcs[2]
per, per_u, per_l = pcs[3]
# a = keplerslaw(per)
# pars = planet.params

dur, dur_l, dur_u = 1.97147557, 1.94595485, 2.00975665
depth, depth_l, depth_u = 969.26349022, 929.73956995, 1008.69414464

r_l, r, r_u = np.percentile(samples.T[0] * np.random.normal(st_r, st_r_e, samples.shape[0])
                            * myc.RS / myc.RE, confs)
a_rs_l, a_rs, a_rs_u = np.percentile(keplerslaw_stellar(samples.T[3], st_m_sample, st_r_sample), confs)
a_au_l, a_au, a_au_u = np.percentile(keplerslaw_stellar(samples.T[3], st_m_sample, st_r_sample) *
                                     np.random.normal(st_r, st_r_e, samples.shape[0]) * myc.RS / myc.AU, confs)
b_l, b, b_u = np.percentile(calc_b(keplerslaw_stellar(samples.T[3], st_m_sample, st_r_sample), samples.T[1]), confs)

s = """$T_{{0}}$ (BJD) & ${:.4f}_{{-{:.4f}}}^{{+{:.4f}}}$ \\\\ [0.1cm]
Period (days) & ${:.4f}_{{-{:.4f}}}^{{+{:.4f}}}$ \\\\ [0.1cm]
$R_{{p}}/R_{{s}}$ & ${:.4f}_{{-{:.4f}}}^{{+{:.4f}}}$ \\\\ [0.1cm]
Inclination (deg.) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
 & \\\\
Derived properties: & \\\\ [0.1cm]
Radius ($R_{{\earth}}$) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
Semi-major axis (sr) & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
Semi-major axis (au) & ${:.4f}_{{-{:.4f}}}^{{+{:.4f}}}$ \\\\ [0.1cm]
Impact parameter & ${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$ \\\\ [0.1cm]
Transit depth (ppm) & ${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$ \\\\ [0.1cm]
Duration (hours) & ${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$ \\\\ [0.1cm]"""\
    .format(t0, t0_l, t0_u, per, per_l, per_u, rp, rp_l, rp_u, inc, inc_l, inc_u,
            r, r-r_l, r_u-r, a_rs, a_rs-a_rs_l, a_rs_u-a_rs, a_au, a_au-a_au_l, a_au_u-a_au, b, b-b_l, b_u-b,
            depth, depth-depth_l, depth_u-depth, dur, dur-dur_l, dur_u-dur)

print s

# # corner plot the MCMC samples
# corner.corner(samples, labels=planet.labels, bins=50, use_math_text=True, plot_contours=True,   # fig=corner_fig,
#               label_kwargs={"fontsize": 20}, hist2d_kwargs={"bins": 50})
# plt.savefig("new_samples.pdf", format="pdf")
# # plt.close("all")
# plt.show()
#
# import sys; sys.exit()


t, f, e = data.LCtime, data.LC, data.LCerror
a = keplerslaw(per)

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

# plt.plot(t, f, "k.")
# plt.plot(t, m)
# plt.show()

nt = 20000  # 20000      # number of phase points
ns = 2000   # 50000      # number of samples of model
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

    depths[i] = (1. - batman.TransitModel(params_rs, np.array([params_rs.t0]), supersample_factor=15,
                                          exp_time=29.4 / 60. / 24.0).light_curve(params_rs)[0]) * 1e6
    inds_in_transit = np.argwhere(m_rs[i] != 1.)
    durations[i] = float(inds_in_transit[-1] - inds_in_transit[0]) * (t_ss[-1] - t_ss[0]) / nt * 24.

print np.percentile(depths, confs)
print np.percentile(durations, confs)

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

mphase, mfold = phase_fold(t_ss, batman.TransitModel(params, t_ss, supersample_factor=1, exp_time=29.4/60./24.0)
                           .light_curve(params), per, t0)
mphase = np.asarray(mphase)
mphase = (mphase - 0.5) * per * 24.
ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2, color="g")

for i, pc in enumerate([[50.-68.27/2., 50., 50.+68.27/2.],       # 1-sigma
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
# ax3.plot(phase, resid, 'k.', ms=10, zorder=1)
ax3.errorbar(phase, resid, e, lw=0., marker=".", ms=10, elinewidth=1.5, color="k", zorder=2)
ax3.plot(phase, resid, 'b.', ms=4, zorder=3)
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
plt.savefig("new_fit.pdf", format="pdf")
# plt.show()
# plt.close()
