import dill
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import batman
import tqdm
from exoplanet import phase_fold
sns.set()

# dir = "jackknife/kelvin/"; n = 12
dir = "jackknife/kelvin2/"; n = 26

pars = ["rprs", "inc", "t0", "per"]
v, ue, le = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

for i in range(n):
    with open("{}{}_mcmc.pkl".format(dir, i), "rb") as pf:
        data, planet, samples = dill.load(pf)

    vals = np.asarray(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))).T

    v = np.append(v, vals[0])
    ue = np.append(ue, vals[1])
    le = np.append(le, vals[2])

v = np.asarray([float(q) for q in v])
ue = np.asarray([float(q) for q in ue])
le = np.asarray([float(q) for q in le])

mean_vals = []
std_err = []

for i in range(4):
    mean = np.mean(v[i::4])
    std = np.std(v[i::4])
    interval = np.array([min(v[i::4]), max(v[i::4])])

    upper_vals = v[i::4] + ue[i::4]
    maxval = max(upper_vals)
    lower_vals = v[i::4] - le[i::4]
    minval = min(lower_vals)

    # print pars[i], mean, (mean-minval), (maxval-mean)     # mean, std, np.abs(interval - mean)
    print pars[i], minval, maxval

    mean_vals.append(mean)
    var_sum = np.sum((v[i::4] - mean)**2.)
    std_err.append(np.sqrt(1./(n*(n-1.)) * var_sum))

    # print mean, std_err[i], std

xi, yi = 0, 1

x = v[xi::4]
y = v[yi::4]
xerr = [le[xi::4], ue[xi::4]]
yerr = [le[yi::4], ue[yi::4]]

plt.figure(figsize=(12, 8))
plt.plot(x, y, ".", c="r", alpha=0.5, ms=15)
sns.kdeplot(x, y, n_levels=200, cmap=sns.cubehelix_palette(as_cmap=True, dark=0., light=1., reverse=True), shade=True)
# sns.kdeplot(x, y, n_levels=20, shade=False)
sns.rugplot(x, c="gray")
sns.rugplot(y, vertical=True, c="gray")
plt.xlabel(pars[xi])
plt.ylabel(pars[yi])
plt.tight_layout()
plt.show()

# plt.errorbar(x, y, yerr, xerr, ls="", marker=".", ms=10, alpha=0.5, elinewidth=1.5)
# for x, y, i in zip(x, y, range(n)):
#     plt.annotate(i, xy=(x, y), textcoords='data')
# plt.xlabel(pars[xi])
# plt.ylabel(pars[yi])
# plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    st_r, st_m = 0.456, 0.497
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


t, f = np.loadtxt("01_lc.dat", delimiter=",", unpack=True)
e = np.std(f)
# t_ss = np.linspace(t[0], t[-1], t.size*100)

models = []
pers, t0s = [], []
tf_models = []
phases, ph_models = [], []

for i in tqdm.tqdm(range(n)):
    with open("{}{}_mcmc.pkl".format(dir, i), "rb") as pf:
        data, planet, samples = dill.load(pf)

    rp, inc, per, t0 = planet.rp_01, planet.inc_01, planet.period_01, planet.t0_01
    a = keplerslaw(per)

    pers.append(per)
    t0s.append(t0)

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
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.0)
    m = bat.light_curve(params)
    # p2, mp = phase_fold(t, m, per, t0 + per / 4.)  # phase-folded model

    # ss model
    t_ss = np.linspace(t0-per/20., t0+per/20., 10000)
    bat_ss = batman.TransitModel(params, t_ss, supersample_factor=15, exp_time=29.4 / 60. / 24.0)
    m_ss = bat_ss.light_curve(params)

    # phm, mp = phase_fold(t_ss, m_ss, per, t0)  # phase-folded model

    tf_models.append(m)
    models.append(m_ss)

models = np.asarray(models)
mod_mean = np.average(models, axis=0)
mod_max = np.max(models, axis=0)
mod_min = np.min(models, axis=0)
residuals = f - np.average(np.asarray(tf_models), axis=0)

print np.mean(pers)
print np.mean(t0s)

# plt.errorbar(x=t, y=f, yerr=e, marker=".", ls="")
# plt.plot(t_ss, mod_mean)
# plt.fill_between(t_ss, mod_min, mod_max, where=mod_min < mod_max, alpha=0.5, interpolate=True)
# plt.show()

# per = np.average(pers)
# t0 = np.average(t0s)
# p1, fp = phase_fold(t, f, per, t0)  # phase-folded flux
# p2, mp_mean = phase_fold(t_ss, mod_mean, per, t0)
# _, mp_max = phase_fold(t_ss, mod_max, per, t0)
# _, mp_min = phase_fold(t_ss, mod_min, per, t0)
#
# plt.figure(figsize=(12, 8))
# plt.errorbar(x=p1, y=fp, yerr=e, marker=".", ls="")
# plt.plot(p2, mp_mean, c="k", alpha=0.5)
# plt.fill_between(p2, mp_max, mp_min, where=mp_min < mp_max, alpha=0.6, interpolate=True, facecolor="grey")
# plt.xlim(0.495, 0.505)
# plt.tight_layout()
# plt.show()

p2 = np.linspace(0.45, 0.55, t_ss.size)
p1, fp = phase_fold(t, f, np.mean(pers), np.mean(t0s))  # phase-folded flux
p3, mp = phase_fold(t, residuals, np.mean(pers), np.mean(t0s))

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
ax1.errorbar(x=p1, y=fp, yerr=e, marker=".", ls="")
ax1.plot(p2, mod_mean, c="k", alpha=0.5)
ax1.fill_between(p2, mod_max, mod_min, where=mod_max > mod_min, alpha=0.6, interpolate=True, facecolor="grey")
ax1.set_xlim(0.495, 0.505)
ax1.tick_params(axis='x', labelbottom=False)

ax2.errorbar(x=p3, y=mp, yerr=e, marker=".", ls="")
ax2.axhline(0., c="k", alpha=0.5)
ax2.fill_between(p2, mod_max-mod_mean, mod_min-mod_mean, where=mod_max > mod_min, alpha=0.6, interpolate=True,
                 facecolor="grey")
# ax2.fill_between(p2, 2.*(mod_max-mod_mean), 2.*(mod_min-mod_mean), where=mod_max > mod_min, alpha=0.4, interpolate=True,
#                  facecolor="grey")
ax2.set_xlim(0.495, 0.505)

ax1.get_shared_x_axes().join(ax1, ax2)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# points to mask for jackknife
# plt.figure(figsize=(14, 7))
# t, f, mod, irange = \
#     t[2000:], f[2000:], np.average(np.asarray(tf_models), axis=0)[2000:], range(t.size)[2000:]
# for x, y, i in zip(t, f, irange):
#     plt.annotate(i, xy=(x, y), textcoords='data')
# plt.plot(t, f, ".")
# plt.plot(t, mod, lw=2, alpha=0.8, zorder=2)
# plt.xlim([min(t), max(t)])
# plt.show()

# tmask = [739, 740, 741, 742, 743, 744, 745, 746,
#          1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
#          3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3138]
#
# pmask = ((t-np.mean(t0s)+np.mean(pers)/2.0) / np.mean(pers)) % 1
#
# in_transit_mask = np.array([x in tmask for x in range(t.size)])
#
# with open("in_transit_mask.dat", "w") as df:
#     for i in range(len(in_transit_mask)):
#         df.write("{}\n".format(in_transit_mask[i]))
#
# pmask, in_transit_mask = zip(*sorted(zip(pmask, in_transit_mask)))
# in_transit_mask = np.asarray(in_transit_mask)
#
# p1, fp = phase_fold(t, f, np.mean(pers), np.mean(t0s))  # phase-folded flux
# p2, mp_mean = phase_fold(t_ss, mod_mean, np.mean(pers), np.mean(t0s))
#
# p1 = np.asarray(p1, dtype=float)
# fp = np.asarray(fp, dtype=float)
#
# plt.plot(p1[in_transit_mask], fp[in_transit_mask], 'o', ms=3, alpha=0.7, zorder=0, c="r")
# plt.plot(p1[in_transit_mask == 0], fp[in_transit_mask == 0], 'o', ms=3, alpha=0.7, zorder=1)
# plt.plot(p2, mp_mean, lw=2, alpha=0.8, zorder=2)
# plt.xlim(0.48, 0.52)
# plt.show()

