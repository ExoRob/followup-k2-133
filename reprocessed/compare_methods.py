import numpy as np
import dill
from scipy.signal import medfilt
import matplotlib.ticker as tic
import matplotlib.pyplot as plt
from astropy.io import fits
import batman
import detrend
from scipy import interpolate
import lightkurve as lk
from my_exoplanet import phase_fold
import more_itertools as mit
from itertools import groupby, count
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")


with open("../transit/pars.pkl", "rb") as pf:
    pars = dill.load(pf)


lc = lk.KeplerLightCurveFile.from_fits("ktwo247887989-c13_llc.fits").PDCSAP_FLUX

t_grid = lc.time

# plt.plot(lc.time, lc.flux, "k.")
# _ = [plt.axvline(3011.1 + v, color="r", alpha=0.5) for v in np.arange(-100, 300)*6./24.]
# plt.show()

lc = lc.remove_nans()
lc = lc[lc.quality == 0]
lc = lc[lc.time > 2988.5]
lc = lc[lc.time < 3064.7]
lc_flat, trend = lc.flatten(window_length=501, return_trend=True)
lc_rm, mask = lc_flat.remove_outliers(return_mask=True)     # remove outliers

mtot = np.ones(lc_rm.time.size, float)
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, lc_rm.time, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    mtot += (bat.light_curve(params) - 1.)
# xa, ya = [], []
# for i in range(4, 35, 5):
#     cor = lc_rm.correct(windows=40, bins=i, niters=4)
#     chisq = np.sqrt(np.sum((cor.flux - mtot)**2.))
#
#     print i, chisq
#     xa.append(i)
#     ya.append(chisq)
# plt.plot(xa, ya, color="k")
# plt.show()
# import sys; sys.exit()

# # sff = lk.SFFCorrector()
# n = 100
# for i in range(0, lc_rm.time.size, n):
#     lc2cor = lc_rm[i:i+n]
#     # corrected_lc = sff.correct(time=lc2cor.time, flux=lc2cor.flux,
#     #                            centroid_col=lc2cor.centroid_col,
#     #                            centroid_row=lc2cor.centroid_row,
#     #                            niters=4, windows=1, bins=20)
#     #
#     # ax = sff._plot_rotated_centroids()
#     # lines = ax.lines
#     # plt.clf()
#     # plt.plot(lines[1].get_xdata(), lines[1].get_ydata())
#     # plt.show()
#
#     corrected_lc = lc2cor.correct(windows=1, bins=15, niters=4)
#     # sff_trend = lc2cor.flux / corrected_lc.flux  # SFF trend
#     # plt.plot(lc2cor.time, lc2cor.flux, "k.")
#     # plt.plot(lc2cor.time, sff_trend, "b-")
#     # plt.plot(corrected_lc.time, corrected_lc.flux, "k.")
#
#     if i == 0:
#         lc_split = corrected_lc
#     else:
#         lc_split = lc_split.append(corrected_lc)

# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3) * pars[3].per + pars[3].t0]
# plt.show()

# lc_cor_mask = lc_split

tmask = np.ones(lc_rm.time.size, bool)
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, lc_rm.time, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    tmask &= (bat.light_curve(params) == 1.)
lc_mask = lc_rm[tmask]

# lc_mask1 = lc_mask[lc_mask.time < 3064.7]
# lc_mask2 = lc_mask[lc_mask.time > 3064.7]

# print len(lc_mask.time), sum(lc_mask.time > 3064.7)
# _ = [plt.axvline(v[0], color="k", ls="--", lw=0.9) for v in np.array_split(lc_mask.time, 40)]

# lc_cor_mask1, save1 = lc_mask1.correct(windows=40, bins=15, niters=4)     # SFF correct non-transit LC
# lc_cor_mask2, save2 = lc_mask2.correct(windows=2, bins=15, niters=4)


# plot_rotated_centroids(pack=save1[0][20])
# plt.show()
# plot_normflux_arclength(pack=save1[3][1][4:])
# plt.show()
#
# lc_cor_mask, lc_mask = lc_cor_mask1, lc_mask1
# lc_rm = lc_rm[lc_rm.time < 3064.7]

# lc_cor_mask = lc_cor_mask1.append(lc_cor_mask2)

# for niter in range(15, 30, 5):
#     lc_cor_mask = lc_rm.correct(windows=20, bins=niter, niters=4)
#
#     plt.plot(lc_cor_mask.time, lc_cor_mask.flux, ".", alpha=0.7, label=niter)
# plt.legend()
# plt.show()

# plt.plot(lc.time, lc.flux, ".", ms=5, alpha=0.6, lw=0.8, ls="-")
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3) * pars[3].per + pars[3].t0]
# plt.show()

# lc_cor_mask, save = lc_mask.correct(windows=40, bins=15, niters=4)
#
# sff_trend = lc_mask.flux / lc_cor_mask.flux       # SFF trend
# sff_int = interp1d(lc_mask.time, sff_trend, "linear")     # interpolate to LC with transits
# sff_int_trend = sff_int(lc_rm.time)
# lc_cor_mask = lk.KeplerLightCurve(time=lc_rm.time, flux=lc_rm.flux / sff_int_trend, flux_err=None)
#
# lc_cor, _ = lc_rm.correct(windows=40, bins=15, niters=4)
# lc_cor_zeit, _ = lc_rm.correct(windows=lc_rm.time.size % 200, bins=15, niters=4)
#
# # lc_rm.flux = lc_rm.flux - mtot + 1.
# # lc_cor_mask = lc_rm.correct(windows=40, bins=15, niters=4)
# # lc_cor_mask.flux = lc_cor_mask.flux + mtot - 1.
#
# # plt.plot(lc_cor_mask.time, lc_cor_mask.flux, "k.", ms=3)
# # plt.show()
#
# in_transit = np.where(mtot != 1.)[0]
# consec = [list(group) for group in mit.consecutive_groups(in_transit)]
#
# plt.plot(lc_rm.time, lc_rm.flux, "k.")
# # # # plt.plot(lc_rm.time[tmask==0], lc_rm.flux[tmask==0], "r.")
# plt.plot(lc_rm.time, sff_int_trend, "g")
# _ = [plt.plot(lc_rm.time[cons], sff_int_trend[cons], "r") for cons in consec]
# # # # plt.plot(lc_rm.time[tmask==0], sff_int_trend[tmask==0], "r.")
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3) * pars[3].per + pars[3].t0]
# _ = [plt.axvline(v, color="b", alpha=0.4, lw=2) for v in np.arange(0, 7) * pars[2].per + pars[2].t0]
# plt.show()

# plt.plot(lc_cor_mask.time, lc_cor_mask.flux, ".", alpha=0.7)
# plt.show()

t_mysff, f_mysff, e_mysff = np.loadtxt("final-lc-mySFF.dat", unpack=True)


lc_sff = fits.open("hlsp_k2sff_k2_lightcurve_247887989-c13_kepler_v1_llc.fits")
data_sff = lc_sff[1].data
t_sff, f_sff = data_sff["T"], data_sff.FCOR
mask_sff = np.abs(f_sff - medfilt(f_sff, 75)) < 2. * np.std(f_sff)
mf_sff = medfilt(f_sff[mask_sff], 71)
t_sff, f_sff = t_sff[mask_sff], f_sff[mask_sff] - mf_sff + np.median(mf_sff)
f_sff = f_sff / np.median(f_sff)
e_sff = np.std(f_sff) * np.ones(t_sff.size, float)


lc_ev = fits.open("hlsp_everest_k2_llc_247887989-c13_kepler_v2.0_lc.fits")
data_ev = lc_ev[1].data
t_ev, f_ev = data_ev.TIME, data_ev.FCOR
t_ev, f_ev = t_ev[f_ev > np.median(f_ev) / 1.2], f_ev[f_ev > np.median(f_ev) / 1.2]
mask_ev = np.abs(f_ev - medfilt(f_ev, 75)) < 2. * np.std(f_ev)
mf_ev = medfilt(f_ev[mask_ev], 71)
t_ev, f_ev = t_ev[mask_ev], f_ev[mask_ev] - mf_ev + np.median(mf_ev)
f_ev = f_ev / np.median(f_ev)
e_ev = np.std(f_ev) * np.ones(t_ev.size, float)


# t_k2sc, f_k2sc, e_k2sc = detrend.open_k2sc("EPIC_247887989_mast.fits", None, False)


with open("k2sc-det.pkl", "rb") as pf:
    t_sc, f_sc, e_sc = dill.load(pf)
f_sc = f_sc / np.median(f_sc)

# t_poly, f_poly, e_poly = np.array([]), np.array([]), np.array([])
# transit_inds = [np.argmin(np.abs(lc.time - pars[3].t0 - i*pars[3].per)) for i in range(3)]
# for i, ti in enumerate(transit_inds):
#     nl = [11-8, 05, 6][i]
#     nu = [11, 15, 3][i]
#
#     t = lc.time[ti-nl:ti+nu+1]
#     f = lc.flux[ti-nl:ti+nu+1] / np.nanmedian(lc.flux)
#     mask = np.abs(np.arange(-nl, nu+1)) > 1
#
#     knots = np.arange(t[0], t[-1], 29.4/0.6875/24.)
#     tisp, c, k = interpolate.splrep(t[mask], f[mask], t=knots[1:])
#     sp = interpolate.BSpline(tisp, c, k)(t)
#
#     t_poly = np.append(t_poly, t)
#     f_poly = np.append(f_poly, f - sp + 1.)
#     e_poly = np.append(e_poly, np.ones(t.size, float))
#
#     # plt.plot(t[mask]-t[0], f[mask], "k.")
#     # plt.plot(t[~mask]-t[0], f[~mask], "r.")
#     # plt.plot(t-t[0], sp, "b-")
#     # plt.show()
#
# np.savetxt("lc-poly.dat", np.array([t_poly, f_poly, e_poly*1.4e-04]).T)

# ph, fp = phase_fold(t_poly, f_poly, pars[3].per, pars[3].t0)
# plt.plot(ph, fp, ".", ms=12, ls="", zorder=2, color="k")
# plt.show()

# for vals in [
#             # [t_sff, f_sff, e_sff, "K2SFF"],
#             # [lc_cor_mask.time, lc_cor_mask.flux, lc_cor_mask.flux_err, "SFF-mask"],
#             # [lc_cor.time, lc_cor.flux, lc_cor.flux_err, "SFF"],
#             # [lc_cor_zeit.time, lc_cor_zeit.flux, lc_cor_zeit.flux_err, "SFF-ZEIT"],
#             [t_ev, f_ev, e_ev, "EVEREST"],
#             [t_sc, f_sc, e_sc, "K2SC"],
#             [t_mysff, f_mysff, e_mysff, "SFF"],
#             # [t_k2sc, f_k2sc, e_k2sc, "K2SC"]
#             # [t_poly, f_poly, e_poly, "Poly"]
#             ]:
#
#     t, f, e, lab = vals
#     plt.plot(t, f, ".", ms=5, label=lab, alpha=0.7)
# plt.legend()
# plt.show()


pl = 3
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fgs = []
ind = 0
for vals in [
            # [t_sff, f_sff, e_sff, "K2SFF"],
            # [lc_cor_mask.time, lc_cor_mask.flux, lc_cor_mask.flux_err, "SFF-mask"],
            # [lc_cor.time, lc_cor.flux, lc_cor.flux_err, "SFF"],
            # [lc_cor_zeit.time, lc_cor_zeit.flux, lc_cor_zeit.flux_err, "SFF-ZEIT"],
            [t_ev, f_ev, e_ev, "EVEREST"],
            [t_sc, f_sc, e_sc, "K2SC"],
            [t_mysff, f_mysff, e_mysff, "SFF"],
            # [t_k2sc, f_k2sc, e_k2sc, "K2SC"]
            # [t_poly, f_poly, e_poly, "Poly"]
            ]:

    t, f, e, lab = vals
    # t -= t[0]

    # f_grid = np.zeros(t_grid.size)
    # for i in range(t_grid.size):
    #     ind = np.argmin(np.abs(t - t_grid[i]))
    #     f_grid[i] = f[ind] if (np.abs(t[ind] - t_grid[i]) < 0.1 / 24.) else np.nan
    # fgs.append(f_grid)
    # plt.plot(t_grid, f_grid, ".", ms=6, label=lab, alpha=0.5, lw=0.9, ls="-")

    # if lab == "SFF":
    #     plt.plot(t_grid, fgs[0] - f_grid, ms=6, alpha=0.9, lw=1.1, ls="-")

    # plt.plot(t, f, ".", ms=5, label=lab, alpha=0.6)

    mtot = np.ones(t.size, float)
    m3 = np.ones(t.size, float)
    for i, params in enumerate(pars):
        bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        mtot += (bat.light_curve(params) - 1.)
        if i < 3:
            m3 += (bat.light_curve(params) - 1.)
    res = f - mtot
    # plt.plot(t, res, ".", ms=6, label=lab, alpha=0.5, lw=0.9, ls="-", zorder=2)

    print lab, np.median([np.std(res[i:i + 13]) for i in range(0, len(res), 13)])*1e6, \
        np.median([np.std(f[i:i + 13]) for i in range(0, len(f), 13)])*1e6

    # rol_prec = [np.std(res[i:i + 12]) for i in range(0, len(res), 12)]
    # print lab, np.median(rol_prec)*1e6     # 6-hour CDPP

    # np.savetxt("lc_none-{}.dat".format(lab), np.array([t, f-mtot, np.ones(t.size)*np.median(rol_prec)]).T)
    # np.savetxt("lc_01-{}.dat".format(lab), np.array([t, f-m3, np.ones(t.size)*np.median(rol_prec)]).T)

    # plt.plot(rol_prec, label=lab)

    # plt.plot(t, mtot, lw=3, alpha=0.4, color="b")

    ph, fp = phase_fold(t, f-m3+1., pars[pl].per, pars[pl].t0)
    plt.plot(ph, fp, alpha=1, lw=2, ls="-", zorder=2, color=["c", "y", "r"][ind])
    # plt.plot(ph, fp, ms=[14, 10, 7][ind], ls="", zorder=3, color="k", marker=[".", "*", "s"][ind])
    # plt.plot(ph, fp, ms=[7, 4, 3][ind], label=lab, zorder=4, color=["c", "y", "r"][ind], ls="",
    #          marker=[".", "*", "s"][ind])
    plt.scatter(ph, fp, s=50, label=lab, zorder=4, color=["c", "y", "r"][ind],
                marker=["o", "^", "s"][ind], edgecolors="k", linewidth=2)

    # for pl in range(4):
    #     others = range(4)
    #     others.remove(pl)
    #     mtot = np.ones(t.size, float)
    #     for i in others:
    #         params = pars[i]
    #         bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    #         mtot += (bat.light_curve(params) - 1.)
    #
    #     x = [[0, 0], [0, 1], [1, 0], [1, 1]][pl]
    #     ph, fp = phase_fold(t, f-mtot+1., pars[pl].per, pars[pl].t0)
    #     axes[x[0]][x[1]].plot(ph, fp, ".", ms=6, label=lab, alpha=0.5, lw=0.9, ls="-", zorder=2)
    #     mt = np.linspace(pars[pl].t0 - 0.1 * pars[pl].per, pars[pl].t0 + 0.1 * pars[pl].per, 1000)
    #     bat = batman.TransitModel(pars[pl], mt, supersample_factor=15, exp_time=29.4/60./24.)
    #     m = bat.light_curve(pars[pl])
    #     axes[x[0]][x[1]].plot(np.linspace(0.4, 0.6, 1000), m, lw=5, alpha=0.4, color="grey", zorder=1)

    ind += 1

# for pl in range(4):
#     c = sns.color_palette()[pl]
#     _ = [plt.axvline(v, color=c, alpha=0.3, lw=2) for v in
#          np.arange(0, int((t_grid[-1] - pars[pl].t0) / pars[pl].per)+1) * pars[pl].per + pars[pl].t0]

mt = np.linspace(pars[pl].t0 - 0.01*pars[pl].per, pars[pl].t0 + 0.01*pars[pl].per, 10000)
bat = batman.TransitModel(pars[pl], mt, supersample_factor=15, exp_time=29.4/60./24.)
m = bat.light_curve(pars[pl])
plt.plot(np.linspace(0.49, 0.51, 10000), m, lw=5, alpha=0.8, color="grey", zorder=1)
plt.xlim(0.4965, 0.5035)
plt.ylim(0.9984, 1.0004)
plt.xlabel("Orbital phase", fontsize=18)
plt.ylabel("Normalised flux", fontsize=18)

ax.xaxis.set_minor_locator(tic.MultipleLocator(base=0.0002))
ax.xaxis.set_major_locator(tic.MultipleLocator(base=0.001))
ax.yaxis.set_minor_locator(tic.MultipleLocator(base=0.0001))
ax.yaxis.set_major_locator(tic.MultipleLocator(base=0.0005))
plt.gca().tick_params(axis='both', which='major', labelsize=14)

plt.legend(fontsize=14)
plt.tight_layout(pad=0.)
plt.savefig("detrend-phase.pdf")
plt.show()
