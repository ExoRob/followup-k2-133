import numpy as np
import dill
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import batman
from scipy import interpolate, linalg, stats
import lightkurve as lk
import warnings
from tqdm import tqdm
from my_exoplanet import phase_fold
# import more_itertools as mit
# from itertools import groupby, count
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()


def fit_bspline(time, flux, knotspacing=1.5):
    """
    Fit a B-spline against flux with knots every few days
    :param time: LC time
    :param flux: LC flux
    :param knotspacing: time between knots
    :return: Spline-interpolation
    """
    knots = np.arange(time[0], time[-1], knotspacing)   # knot points

    # If the light curve has breaks larger than the knot-spacing, we must remove the knots that fall in the breaks.
    bad_knots = []
    a = time[0:-1][np.diff(time) > knotspacing]
    b = time[1:][np.diff(time) > knotspacing]
    for a1, b1 in zip(a, b):
        bad = np.where((knots > a1) & (knots < b1))[0][1:-1]
        if len(bad_knots) > 0:
            [bad_knots.append(b) for b in bad]
    good_knots = list(set(list(np.arange(len(knots)))) - set(bad_knots))
    knots = knots[good_knots]

    # Now fit and return the spline
    ti, c, k = interpolate.splrep(time, flux, t=knots[1:])

    return interpolate.BSpline(ti, c, k)


def rotate_centroids(centroid_col, centroid_row):
    """
    Rotate the coordinate frame of the (col, row) centroids to a new (x,y) frame in
    which the dominant motion of the spacecraft is aligned with the x axis.
    This makes it easier to fit a characteristic polynomial that describes the motion.
    """
    centroids = np.array([centroid_col, centroid_row])
    _, eig_vecs = linalg.eigh(np.cov(centroids))

    return np.dot(eig_vecs, centroids)


def bin_and_interpolate(s, normflux, bins, sigma, inmask, plot=False):
    """
    Fit the arclengths vs flux (per window)
    :param s: arclengths
    :param normflux: Flux normalised by window
    :param bins: Number of bins to divide window into
    :param sigma: Sigma to use in outlier removal
    :param inmask: In-transit mask
    :param plot: Plot the fit? [bool]
    :return: Interpolated arclength vs flux & outlier mask
    """
    idx = np.argsort(s)             # order of arclengths
    s_srtd = s[idx]                 # sorted arclengths
    normflux_srtd = normflux[idx]   # sorted flux

    # remove outliers
    outlier_mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask | ~inmask   # outlier & transit mask
    normflux_srtd = normflux_srtd[~outlier_mask]
    s_srtd = s_srtd[~outlier_mask]

    knots = np.array([np.min(s_srtd)]
                     + [np.median(split) for split in np.array_split(s_srtd, bins)]
                     + [np.max(s_srtd)])
    bin_means = np.array([normflux_srtd[0]]
                         + [np.mean(split) for split in np.array_split(normflux_srtd, bins)]
                         + [normflux_srtd[-1]])

    if plot:
        plt.plot(s_srtd, normflux_srtd, ".")
        plt.plot(knots, bin_means, "o")
        plt.show()

    return interpolate.interp1d(knots, bin_means, bounds_error=False, fill_value='extrapolate'), outlier_mask


def arclength(x1, x_full, polyp):
    """
    Compute the arclength of the polynomial used to fit the centroid measurements
    :param x1: Integrate to here
    :param x_full: Full arc
    :param polyp: Polynomial of rotated (x,y) fit
    :return: Arclength up to x1
    """
    msk = x_full < x1   # arc up to x1

    return np.trapz(y=np.sqrt(1. + polyp(x_full[msk]) ** 2.), x=x_full[msk])


def correct(time, flux, centroid_col, centroid_row, breaks, tr_mask=None, polyorder=5, niters=3, bins=15, sigma_1=3.,
            sigma_2=5., restore_trend=False, plot_arcs=False, plot_windows=False):
    """
    Implements the Self-Flat-Fielding (SFF) systematics removal method. See Vanderburg and Johnson (2014).

    Briefly, the algorithm can be described as follows
       (1) Rotate the centroid measurements onto the subspace spanned by the eigenvectors of the centroid covariance
           matrix
       (2) Fit a polynomial to the rotated centroids
       (3) Compute the arclength of such polynomial
       (4) Fit a BSpline of the raw flux as a function of time
       (5) Normalize the raw flux by the fitted BSpline computed in step (4)
       (6) Bin and interpolate the normalized flux as a function of the arclength
       (7) Divide the raw flux by the piecewise linear interpolation done in step (6)
       (8) Set raw flux as the flux computed in step (7) and repeat
       (9) Multiply back the fitted BSpline

    :param time: LC time
    :param flux: LC flux
    :param centroid_col: Detector x position
    :param centroid_row: Detector y position
    :param breaks: Indices to split the LC at
    :param tr_mask: Transit mask
    :param polyorder: Order of polynomial fit to rotated centroids
    :param niters: number of times to run algorithm
    :param bins: number of bins for arclength-flux interpolation
    :param sigma_1: outlier sigma in flux
    :param sigma_2: outlier sigma in centroid fit
    :param restore_trend: add the long-term trend back in
    :param plot_arcs: plot arclength vs flux for whole iteration [bool]
    :param plot_windows: plot centroids, rotated centroids and arclength vs flux for each window [bool]
    :return: arclength-corrected flux & trend
    """
    assert not (plot_arcs and plot_windows), "Can only plot one ..."

    windows = breaks.size + 1   # number of windows from break-points
    pbar = tqdm(total=niters * windows, initial=0, desc="Running SFF")  # total progress bar (windows X niters)

    timecopy = np.copy(time)

    col_lims = (centroid_col.min(), centroid_col.max())
    row_lims = (centroid_row.min(), centroid_row.max())
    f_lims = (f.min(), f.max())

    if tr_mask is None:
        tr_mask = np.ones(t.size, bool)

    # split into windows
    time = np.split(time, breaks)
    flux = np.split(flux, breaks)
    centroid_col = np.split(centroid_col, breaks)
    centroid_row = np.split(centroid_row, breaks)
    trend = np.split(np.ones(timecopy.size, float), breaks)  # flux correction trend (f / f-hat)
    tr_mask = np.split(tr_mask, breaks)

    # apply the correction iteratively
    for n in range(niters):
        tempflux = np.asarray([item for sublist in flux for item in sublist])
        flux_outliers = sigma_clip(data=tempflux, sigma=sigma_1).mask | \
            ~np.asarray([item for sublist in tr_mask for item in sublist])
        bspline = fit_bspline(timecopy[~flux_outliers], tempflux[~flux_outliers], knotspacing=3.)
        # plt.plot(timecopy[~flux_outliers], tempflux[~flux_outliers], "k.")
        # plt.plot(timecopy[flux_outliers], tempflux[flux_outliers], "r.")
        # plt.plot(timecopy, bspline(timecopy))
        # plt.show()

        if plot_arcs:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # SFF algorithm is run on each window independently
        for i in range(windows):
            # To make it easier (and more numerically stable) to fit a characteristic polynomial that describes the
            # spacecraft motion, we rotate the centroids to a new coordinate frame in which the dominant direction of
            # motion is aligned with the x-axis.
            rot_col, rot_row = rotate_centroids(centroid_col[i], centroid_row[i])

            # Next, we fit the motion polynomial after removing outliers
            outlier_cent = sigma_clip(data=rot_col, sigma=sigma_2).mask | ~tr_mask[i]

            with warnings.catch_warnings():
                # ignore warning messages related to polyfit being poorly conditioned
                warnings.simplefilter("ignore", category=np.RankWarning)
                coeffs = np.polyfit(rot_row[~outlier_cent], rot_col[~outlier_cent], polyorder)

            poly = np.poly1d(coeffs)
            polyprime = poly.deriv()

            # Compute the arclengths. It is the length of the polynomial that describes the typical motion.
            x = np.linspace(np.min(rot_row[~outlier_cent]), np.max(rot_row[~outlier_cent]), 10000)
            s = np.array([arclength(x1=xp, x_full=x, polyp=polyprime) for xp in rot_row])

            # Remove the long-term variation by dividing the flux by the spline
            iter_trend = bspline(time[i])
            normflux = flux[i] / iter_trend
            trend[i] *= iter_trend

            # normflux = flux[i]

            # Bin and interpolate normalized flux to capture the dependency of the flux as a function of arclength
            interp, outlier_mask = bin_and_interpolate(s, normflux, bins, sigma=sigma_1, inmask=tr_mask[i])

            # Correct the raw flux
            corrected_flux = normflux / interp(s)
            trend[i] = np.asarray(interp(s)) * np.asarray(trend[i])
            flux[i] = corrected_flux
            if restore_trend:
                flux[i] *= trend[i]

            idx = np.argsort(s)
            s_srtd = s[idx]
            normflux_srtd = normflux[idx]

            # save[n][i] = [rot_col, rot_row, outlier_cent, poly, s, normflux, interp, outlier_mask]

            if plot_windows:
                plt.plot(centroid_col[i], centroid_row[i], 'ko', ms=4)
                plt.plot(centroid_col[i], centroid_row[i], 'bo', ms=1)
                plt.xlim(col_lims)
                plt.ylim(row_lims)
                plt.savefig("plots/centroids-{}-{}.pdf".format(n, i))
                plt.clf()

                # print min(rot_row), max(rot_row)
                # print min(rot_col), max(rot_col)

                plt.plot(rot_row[~outlier_cent], rot_col[~outlier_cent], 'ko', markersize=4)
                plt.plot(rot_row[~outlier_cent], rot_col[~outlier_cent], 'bo', markersize=1)
                plt.plot(rot_row[outlier_cent], rot_col[outlier_cent], 'ko', markersize=4)
                plt.plot(rot_row[outlier_cent], rot_col[outlier_cent], 'ro', markersize=1)
                x = np.linspace(min(rot_row), max(rot_row), 1000)
                plt.plot(x, poly(x), '--')
                plt.xlabel("Rotated row centroid")
                plt.ylabel("Rotated column centroid")
                plt.savefig("plots/rot-centroids-{}-{}.pdf".format(n, i))
                plt.clf()

                plt.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'ko', markersize=4)
                plt.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'bo', markersize=1)
                plt.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ko', markersize=4)
                plt.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ro', markersize=1)
                plt.plot(s_srtd, interp(s_srtd), '--')
                plt.xlim(-0.5, 5.)
                plt.ylim(f_lims)
                plt.xlabel(r"Arclength $(s)$")
                plt.ylabel(r"Flux $(e^{-}s^{-1})$")
                plt.savefig("plots/arclengths-{}-{}.pdf".format(n, i))
                plt.clf()

            if plot_arcs:
                ax.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'ko', markersize=3, zorder=1)
                ax.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'bo', markersize=2, zorder=2)
                ax.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ko', markersize=3, zorder=1)
                ax.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ro', markersize=2, zorder=2)
                ax.plot(s_srtd, interp(s_srtd), '-', color="g", lw=2, zorder=3, alpha=0.8)

            pbar.update(1)
        if plot_arcs:
            plt.xlabel(r"Arclength $(s)$")
            plt.ylabel(r"Flux $(e^{-}s^{-1})$")
            plt.show()

    flux_hat = np.asarray([item for sublist in flux for item in sublist])
    pbar.close()

    return flux_hat, np.asarray([item for sublist in trend for item in sublist])  # , save


# load the light curve
lc = fits.open("ktwo247887989-c13_llc.fits")[1].data
t = lc.TIME
f = lc.PDCSAP_FLUX
e = lc.PDCSAP_FLUX_ERR
q = lc.SAP_QUALITY

# x = lc.POS_CORR1
# y = lc.POS_CORR2
x = lc.PSF_CENTR1 - np.nanmean(lc.PSF_CENTR1)
y = lc.PSF_CENTR2 - np.nanmean(lc.PSF_CENTR2)
# x = lc.MOM_CENTR1 - np.nanmean(lc.MOM_CENTR1)
# y = lc.MOM_CENTR2 - np.nanmean(lc.MOM_CENTR2)

# for vals in [[lc.POS_CORR1, lc.POS_CORR2, "COR"],
#              [lc.PSF_CENTR1 - np.nanmean(lc.PSF_CENTR1), lc.PSF_CENTR2 - np.nanmean(lc.PSF_CENTR2), "PSF"],
#              [lc.MOM_CENTR1 - np.nanmean(lc.MOM_CENTR1), lc.MOM_CENTR2 - np.nanmean(lc.MOM_CENTR2), "MOM"]]:
#     x, y, lab = vals
#
#     mask = np.isfinite(t * f * e * x * y) & (q == 0)  # nan and quality mask
#     mask &= t > 2988.5
#     mask &= t < 3064.7
#
#     # plt.plot(x[mask], y[mask], ".", ms=3, label=lab)
#     plt.plot(t[mask], x[mask], ".", ms=3, label=lab+"x")
#     plt.plot(t[mask], y[mask], ".", ms=3, label=lab+"y")
# plt.legend()
# plt.show()

mask = np.isfinite(t * f * e * x * y) & (q == 0)    # nan and quality mask
mask &= t > 2988.5
mask &= t < 3064.7
# mask &= np.arange(t.size) > 197

with open("../transit/pars.pkl", "rb") as pf:
    pars = dill.load(pf)
models = []
m_all, m_4 = np.ones(t.size, float), np.ones(t.size, float)
tmask = np.ones(t.size, bool)
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.)
    m = bat.light_curve(params)
    models.append(m)
    tmask &= (m == 1.)

    if i < 3:
        m_all += (m - 1.)
    m_4 += (m - 1.)

# find outliers
t_ok = t[mask]
f_ok = f[mask]
f_ok /= np.median(f_ok)
rm = median_filter(f_ok, 71)     # running median
omask = (f_ok - rm) < (np.std(f_ok - rm + 1.)*5.)   # outlier mask

t = t[mask][omask]
f = f[mask][omask]
e = e[mask][omask]
x = x[mask][omask]
y = y[mask][omask]
q = q[mask][omask]
tmask = tmask[mask][omask]
m_all = m_all[mask][omask]
m_4 = m_4[mask][omask]

# remove long-term trend
# mf = median_filter(f, 72*2)
#
# tfit, ffit, nout, nitr = t, f, 1, 0
# while nout > 0:
#     bs = fit_bspline(tfit, ffit, knotspacing=3.)
#     bsfit = bs(tfit)
#     outliers = sigma_clip(ffit - bsfit, 3.).mask
#     nout = outliers.sum()
#     tfit, ffit = tfit[~outliers], ffit[~outliers]
#     nitr += 1
# bs = bs(t)
#
# # plt.plot(t, f, "k.", label="Data")
# # plt.plot(t, mf, label="Median filter")
# # plt.plot(t, bs, label="Spline")
# # plt.show()
#
# # f = f - mf + np.median(mf)
# f = f - bs + np.median(bs)

f /= np.median(f)

splits = [2997., 3033.]     # campaign 13

# rang = np.arange(1955-378-50, 1955-378+50)
# for j in range(170, 210):
#     rems = rang % j
#     if any(rems == 0):
#         print rang[rems == 0], j, np.abs(1955-378-rang[rems == 0])

break_points = np.arange(0, t.size, 197)[1:]

# print [len(sec) for sec in np.split(t, break_points)]
# print t.size - break_points[-1]

# tsplit, fsplit = np.split(t, break_points), np.split(f, break_points)
# for i in range(len(break_points)):
#     plt.plot(tsplit[i], fsplit[i], ".")
# plt.show()

# plt.plot(t, x, ".")
# plt.plot(t, y, ".")
# _ = [plt.axvline(t[v], color="0.8") for v in break_points]
# plt.show()

fc, tr = correct(t, f, x*4., y*4., break_points, np.ones(tmask.shape, bool), niters=2,
                 # plot_windows=True,
                 # plot_arcs=True
                 )

# _ = [plt.axvline(v, color="0.8", ls="--") for v in splits]
# plt.plot(t, f, "ko", ms=1)
# plt.plot(t, tr)
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
# plt.show()

# _ = [plt.axvline(v, color="0.8", ls="--") for v in splits]
# plt.plot(t, fc, "ko", ms=1)
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
# plt.show()

res = fc - m_4
err = np.median([np.std(res[i:i + 12]) for i in range(0, len(res), 12)]) * np.ones(t.size, float)
print err[0]*1e6

# np.savetxt("lc-mySFF.dat", np.array([t, fc, err]).T)
# np.savetxt("lc-01-mySFF.dat", np.array([t, fc-m_all+1., err]).T)
# np.savetxt("lc-none-mySFF.dat", np.array([t, fc-m_4+1., err]).T)

# mtot = np.ones(t.size, float)
# for i in range(3):
#     params = pars[i]
#     bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
#     mtot += (bat.light_curve(params) - 1.)
# phase = (t - pars[3].t0 + pars[3].per/2.) / pars[3].per % 1
# ctmask = (phase >= 0.495) & (phase <= 0.505)
# plt.plot(t[ctmask], fc[ctmask]-mtot[ctmask]+1., "k.")
# plt.plot(t[~ctmask], fc[~ctmask]-mtot[~ctmask]+1., "r.")
# plt.show()
#
# plt.plot(t, e/np.nanmedian(lc.PDCSAP_FLUX))
# plt.axhline(err[0])
# plt.show()
# np.savetxt("lc-mySFF-cut-transits.dat", np.array([t[ctmask], fc[ctmask]-mtot[ctmask]+1., err[ctmask]]).T)

# plt.plot(t, res, "k.")
# _ = [plt.axhline(ev, alpha=0.6) for ev in np.arange(-5, 5)*err[0]]
# plt.axhline(-1e-3)
# plt.show()

# sns.distplot(res, bins=150, fit=stats.norm, rug=True, kde=False)
# plt.show()

# print sum(res == 0.), break_points.size

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for pl in range(4):
    others = range(4)
    others.remove(pl)
    mtot = np.ones(t.size, float)
    for i in others:
        params = pars[i]
        bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        mtot += (bat.light_curve(params) - 1.)

    x = [[0, 0], [0, 1], [1, 0], [1, 1]][pl]
    ph, fp = phase_fold(t, fc - mtot + 1., pars[pl].per, pars[pl].t0)

    # axes[x[0]][x[1]].plot(ph, fp, "k.", ms=8, zorder=2)
    axes[x[0]][x[1]].errorbar(ph, fp, err, lw=0., marker=".", ms=8, elinewidth=1.5, color="k", zorder=2)
    axes[x[0]][x[1]].plot(ph, fp, ".", ms=4, lw=0.9, ls="-", zorder=3, color="b")

    mt = np.linspace(pars[pl].t0 - 0.1 * pars[pl].per, pars[pl].t0 + 0.1 * pars[pl].per, 1000)
    bat = batman.TransitModel(pars[pl], mt, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    m = bat.light_curve(pars[pl])
    axes[x[0]][x[1]].plot(np.linspace(0.4, 0.6, 1000), m, lw=4, alpha=0.7, color="grey", zorder=1)

    pho = [0.025, 0.015, 0.015, 0.005][pl]
    axes[x[0]][x[1]].set_xlim(0.5-pho, 0.5+pho)
    axes[x[0]][x[1]].set_ylim(0.9978, 1.0008)
plt.legend()
plt.tight_layout(pad=0.)
plt.savefig("original.pdf")
plt.show()
