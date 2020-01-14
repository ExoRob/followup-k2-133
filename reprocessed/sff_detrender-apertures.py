# from __future__ import print_function
import numpy as np
import dill
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import batman
from scipy import interpolate, linalg, stats
# from scipy.optimize import curve_fit
import matplotlib
from matplotlib.colors import LogNorm
from lightkurve import KeplerTargetPixelFile
from scipy.stats import norm, poisson
import warnings
from tqdm import tqdm
import sys
sys.path.append("/Users/rwells/MCcubed/")
import MCcubed as mc3
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")


def phase_fold(in_time, in_flux, period, tt0):
    """
    Phase fold a LC with a given period and epoch
    :param in_time: LC time
    :param in_flux: LC flux
    :param period: Orbital period
    :param tt0: Transit epoch
    :return: phase centred on 0.5, flux vs phase
    """
    foldtimes = ((in_time - tt0 + period / 2.0) / period) % 1

    phase, flux_srtd = zip(*sorted(zip(foldtimes, in_flux)))

    return np.asarray(phase), np.asarray(flux_srtd)


def fit_bspline(time, flux, knotspacing=1.5, extrap=False, slim=None):
    """
    Fit a B-spline against flux with knots every few days
    :param time: LC time
    :param flux: LC flux
    :param knotspacing: time between knots
    :param extrap: extrapolate spline?
    :param slim: smoothing condition
    :return: Spline-interpolation
    """
    knots = np.arange(time.min(), time.max(), knotspacing)   # knot points

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
    ti, c, k = interpolate.splrep(time, flux, t=knots[1:], s=slim, w=np.ones(time.size, float))

    return interpolate.BSpline(ti, c, k, extrapolate=extrap)


def fit_bspline_recursive(tfit, ffit, knotspacing=1.5, sigma=3., outliers=None, extrap=False, slim=None):
    """
    Iteratively fit a B-spline
    :param tfit: x-axis for fit
    :param ffit: y-axis for fit
    :param knotspacing: space between knot-points
    :param sigma: sigma-clipping condition
    :param outliers: initial outliers for fit
    :param extrap: extrapolate spline?
    :param slim: smoothing condition
    :return: Spline-interpolation & outliers
    """
    if outliers is None:    # initial outliers, e.g. in-transit
        outliers = np.zeros(tfit.size, bool)    # 1=outlier
    nout, nitr = 0, 0       # no. of outliers, no. of iterations
    tcopy = tfit
    inds = np.arange(tcopy.size)

    while nout > 0 or nitr == 0:     # while new outliers
        tfit, ffit = tfit[~outliers], ffit[~outliers]           # remove any outliers
        inds = inds[~outliers]                                  # recover which value

        bs = fit_bspline(tfit, ffit, knotspacing=knotspacing, extrap=extrap, slim=slim)   # fit spline
        bsfit = bs(tfit, extrapolate=extrap)

        outliers = sigma_clip(ffit - bsfit, sigma).mask         # find outliers
        nout = outliers.sum()

        nitr += 1

    outs = ~np.array([ind in inds for ind in np.arange(tcopy.size)])    # outlier=1

    return bs, outs


def rotate_centroids(centroid_col, centroid_row):
    """
    Rotate the coordinate frame of the (col, row) centroids to a new (x,y) frame in
    which the dominant motion of the spacecraft is aligned with the x axis.
    This makes it easier to fit a characteristic polynomial that describes the motion.
    """
    centroids = np.array([centroid_col, centroid_row])
    _, eig_vecs = linalg.eigh(np.cov(centroids))

    return np.dot(eig_vecs, centroids)


def arclength(x1, x_full, polyp):
    """
    Compute the arclength of the polynomial used to fit the centroid measurements
    :param x1: Integrate to here
    :param x_full: Full arc
    :param polyp: Differentiated polynomial of rotated (x,y) fit
    :return: Arclength up to x1
    """
    msk = x_full < x1   # arc up to x1

    return np.trapz(y=np.sqrt(1. + polyp(x_full[msk]) ** 2.), x=x_full[msk])


def arclength_spline(x1, x2, spline):
    """
    Same as 'arclength' function but for when using splines instead of polynomials
    :param x1: Integrate from here
    :param x2: Integrate to here
    :param spline: rotated x-y fitted spline
    :return: Arclength of spline
    """
    x_full = np.linspace(x1, x2, 1000)
    y_full = spline.derivative()(x_full)

    return np.trapz(y=np.sqrt(1. + y_full**2.), x=x_full)


def correct(time, flux, centroid_col, centroid_row, breaks, tr_mask=None, method="poly", polyorder=5, niters=3,
            sigma_1=3., sigma_2=5., progress=True, restore_trend=False, plot_arcs=False, plot_windows=False,
            time_knotspacing=3., rot_knotspacing=0.689, arc_knotspacing=0.5):
    """
    Implements the Self-Flat-Fielding (SFF) systematics removal method. See Vanderburg and Johnson (2014).

    Briefly, the algorithm can be described as follows
       (1) Rotate the centroids onto the subspace spanned by the eigenvectors of the centroid covariance matrix
       (2) Fit a polynomial to the rotated centroids
       (3) Compute the arclength of such polynomial
       (4) Fit a BSpline of the raw flux as a function of time
       (5) Normalize the raw flux by the fitted BSpline computed in step (4)
       (6) Interpolate the normalized flux as a function of the arclength
       (7) Divide the raw flux by the piecewise linear interpolation done in step (6)
       (8) Set raw flux as the flux computed in step (7) and repeat
       (9) Multiply back the fitted BSpline

    :param time: LC time
    :param flux: LC flux
    :param centroid_col: Detector x position
    :param centroid_row: Detector y position
    :param breaks: Indices to split the LC at
    :param tr_mask: Transit mask
    :param method: poly or spline fit to centroids?
    :param polyorder: Order of polynomial fit to rotated centroids
    :param niters: number of times to run algorithm
    :param sigma_1: outlier sigma in flux
    :param sigma_2: outlier sigma in centroid fit
    :param progress: show progress bar [bool]
    :param restore_trend: add the long-term trend back in
    :param plot_arcs: plot arclength vs flux for whole iteration [bool]
    :param plot_windows: plot centroids, rotated centroids and arclength vs flux for each window [bool]
    :param time_knotspacing: spline knot-spacing for long-term time-flux trend
    :param rot_knotspacing: spline knot-spacing for rotated centroids fit
    :param arc_knotspacing: spline knot-spacing for arclength-flux trend
    :return: arclength-corrected flux & trend
    """
    assert not (plot_arcs and plot_windows), "Can only plot one ..."

    windows = breaks.size + 1   # number of windows from break-points

    if progress:
        pbar = tqdm(total=niters * windows, initial=0, desc="Running SFF")  # total progress bar (windows X niters)

    timecopy = np.copy(time)    # full LC time

    # find limits to plot each window on the same scale
    col_lims = (centroid_col.min(), centroid_col.max())
    row_lims = (centroid_row.min(), centroid_row.max())
    f_lims = (flux.min(), flux.max())

    if tr_mask is None:
        tr_mask = np.ones(timecopy.size, bool)

    # split into windows by breakpoints
    time = np.split(time, breaks)                               # time
    flux = np.split(flux, breaks)                               # flux
    centroid_col = np.split(centroid_col, breaks)               # x
    centroid_row = np.split(centroid_row, breaks)               # y
    trend = np.split(np.ones(timecopy.size, float), breaks)     # flux correction trend (f / f-hat)
    tr_mask = np.split(tr_mask, breaks)                         # out-of-transit mask
    s_array = np.split(np.zeros(timecopy.size, float), breaks)  # arclengths

    spl_trend = np.split(np.ones(timecopy.size, float), breaks)
    arc_trend = np.split(np.ones(timecopy.size, float), breaks)

    # apply the correction iteratively
    for n in range(niters):
        tempflux = np.asarray([item for sublist in flux for item in sublist])   # full LC flux after n iterations

        # fit B-spline to entire LC
        bspline, flux_outliers = fit_bspline_recursive(timecopy, tempflux, time_knotspacing, sigma_1,
                                                       ~np.asarray([item for sublist in tr_mask for item in sublist]),
                                                       extrap=True)
        # outliers = np.split(flux_outliers, breaks)

        # plt.plot(timecopy[~flux_outliers], tempflux[~flux_outliers], "k.")
        # plt.plot(timecopy[flux_outliers], tempflux[flux_outliers], "r.")
        # plt.plot(timecopy, bspline(timecopy))
        # _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3) * pars[3].per + pars[3].t0]
        # plt.show()

        if plot_arcs:   # plot (x, y) arcs for each iteration?
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # SFF algorithm is run on each window independently
        for i in range(windows):
            # To make it easier (and more numerically stable) to fit a characteristic polynomial that describes the
            # spacecraft motion, we rotate the centroids to a new coordinate frame in which the dominant direction of
            # motion is aligned with the x-axis.

            # # find windows of the candidate planet
            # if (batman.TransitModel(pars[3], time[i], supersample_factor=15, exp_time=29.4/60./24.).
            #         light_curve(pars[3]) != 1.).sum() > 0:
            #     # print i
            #     plot_windows = True
            # else:
            #     plot_windows = False

            # centroid_col[i], centroid_row[i] = \
            #     centroid_col[i] - centroid_col[i].mean(), centroid_row[i] - centroid_row[i].mean()

            rot_col, rot_row = rotate_centroids(centroid_col[i], centroid_row[i])   # rotate (x, y) frame
            # rot_col, rot_row = centroid_col[i], centroid_row[i]

            # find outliers or where in-transit     TODO: do (x, y) outliers exist?
            outlier_cent = sigma_clip(data=rot_col, sigma=sigma_2).mask | ~tr_mask[i]

            # plt.plot(rot_col[~outlier_cent], rot_row[~outlier_cent], "k.")
            # plt.plot(rot_col[outlier_cent], rot_row[outlier_cent], "r.")
            # plt.show()

            # fit poly/spline to rotated centroids
            if method == "poly":
                x = np.linspace(np.min(rot_row[~outlier_cent]), np.max(rot_row[~outlier_cent]), 1000)  # super-sample x
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.RankWarning)    # ignore polyfit warning messages
                    # fit polynomial to rotated centroids (arc)
                    coeffs = np.polyfit(rot_row[~outlier_cent], rot_col[~outlier_cent], polyorder)

                poly = np.poly1d(coeffs)    # polynomial

                # Compute the arclengths. It is the length of the polynomial that describes the typical motion.
                s = np.array([arclength(x1=xp, x_full=x, polyp=poly.deriv()) for xp in rot_row])     # calc arclengths

            else:   # spline method
                srt = np.argsort(rot_row)
                rot_row_srtd = rot_row[srt]
                rot_col_srtd = rot_col[srt]
                outlier_cent = outlier_cent[srt]
                rot_spline, _ = fit_bspline_recursive(rot_row_srtd[~outlier_cent], rot_col_srtd[~outlier_cent],
                                                      knotspacing=rot_knotspacing, sigma=sigma_2, extrap=True)
                s_spline = np.array([arclength_spline(x1=np.min(rot_row[~outlier_cent]),
                                                      x2=xp, spline=rot_spline) for xp in rot_row])
                # plt.plot(rot_row, s, ".")
                # plt.plot(rot_row, s_spline, ".")
                # plt.show()
                s = s_spline.copy()

            s_array[i] = s.copy()

            # Remove the long-term flux trend
            iter_trend = bspline(time[i])
            normflux = flux[i] - iter_trend + 1.
            trend[i] += (iter_trend - 1.)

            spl_trend[i] += (iter_trend - 1.)

            # Interpolate normalized flux to capture the dependency of the flux as a function of arclength
            idx = np.argsort(s)
            s_srtd = s[idx]
            normflux_srtd = normflux[idx]

            interp, outlier_mask = fit_bspline_recursive(s_srtd, normflux_srtd, knotspacing=arc_knotspacing,
                                                         sigma=sigma_1, outliers=~tr_mask[i], extrap=True)
            # plt.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], "k.")
            # plt.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], "r.")
            # plt.plot(s_srtd, interp(s_srtd), "b-")
            # plt.show()

            # Correct the raw flux
            corrected_flux = normflux - interp(s) + 1.
            trend[i] = trend[i] + np.asarray(interp(s)) - 1.

            arc_trend[i] += (np.asarray(interp(s)) - 1.)

            flux[i] = corrected_flux
            if restore_trend:
                flux[i] = flux[i] - trend[i] + 1.

            if plot_windows:
                plt.plot(centroid_col[i], centroid_row[i], 'ko', ms=4)
                plt.plot(centroid_col[i], centroid_row[i], 'bo', ms=1)
                plt.xlim(col_lims)
                plt.ylim(row_lims)
                # plt.savefig("plots/centroids-{}-{}.pdf".format(n, i))
                # plt.clf()
                plt.show()

                # print min(rot_row), max(rot_row)
                # print min(rot_col), max(rot_col)

                plt.plot(rot_row[~outlier_cent], rot_col[~outlier_cent], 'ko', markersize=4)
                plt.plot(rot_row[~outlier_cent], rot_col[~outlier_cent], 'bo', markersize=1)
                plt.plot(rot_row[outlier_cent], rot_col[outlier_cent], 'ko', markersize=4)
                plt.plot(rot_row[outlier_cent], rot_col[outlier_cent], 'ro', markersize=1)
                x = np.linspace(min(rot_row), max(rot_row), 1000)
                if method == "poly":
                    plt.plot(x, poly(x), '--')
                else:
                    plt.plot(x, rot_spline(x), "--")
                plt.xlabel("Rotated row centroid")
                plt.ylabel("Rotated column centroid")
                # plt.savefig("plots/rot-centroids-{}-{}.pdf".format(n, i))
                # plt.clf()
                plt.show()

                plt.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'ko', markersize=4)
                plt.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'bo', markersize=1)
                plt.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ko', markersize=4)
                plt.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ro', markersize=1)
                plt.plot(s_srtd, interp(s_srtd), '--')
                plt.xlim(-0.5, 5.)
                plt.ylim(f_lims)
                plt.xlabel(r"Arclength $(s)$")
                plt.ylabel(r"Flux $(e^{-}s^{-1})$")
                # plt.savefig("plots/arclengths-{}-{}.pdf".format(n, i))
                # plt.clf()
                plt.show()

            if plot_arcs:
                ax.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'ko', markersize=3, zorder=1)
                ax.plot(s_srtd[~outlier_mask], normflux_srtd[~outlier_mask], 'bo', markersize=2, zorder=2)
                ax.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ko', markersize=3, zorder=1)
                ax.plot(s_srtd[outlier_mask], normflux_srtd[outlier_mask], 'ro', markersize=2, zorder=2)
                ax.plot(s_srtd, interp(s_srtd), '-', color="g", lw=2, zorder=3, alpha=0.8)

            if progress:
                pbar.update(1)
        if plot_arcs:
            plt.xlabel(r"Arclength $(s)$")
            plt.ylabel(r"Flux $(e^{-}s^{-1})$")
            plt.show()

    flux_hat = np.asarray([item for sublist in flux for item in sublist])

    if progress:
        pbar.close()

    plt.close("all")

    return flux_hat, np.asarray([item for sublist in trend for item in sublist]), \
        np.asarray([item for sublist in s_array for item in sublist]), \
           np.asarray([item for sublist in spl_trend for item in sublist]), \
           np.asarray([item for sublist in arc_trend for item in sublist])


def plot_contours(image, ax):
    image = image
    x = np.arange(0, len(image[0]))
    y = np.arange(0, len(image))

    px, py = [], []

    # edges first
    for i in x:
        if image[0][i] == 1:
            px += [i, i+1]
            py += [0, 0]
            ax.plot([i, i+1], [0, 0], color="g", lw=2)
        if image[y[-1]][i] == 1:
            px += [i, i+1]
            py += [len(y), len(y)]
            ax.plot([i, i+1], [len(y), len(y)], color="g", lw=2)

    for j in y:
        if image[j][0] == 1:
            px += [0, 0]
            py += [j, j+1]
            ax.plot([0, 0], [j, j+1], color="g", lw=2)
        if image[j][x[-1]] == 1:
            px += [len(x), len(x)]
            py += [j, j+1]
            ax.plot([len(x), len(x)], [j, j+1], color="g", lw=2)

    # middle
    for j in y:
        for i in x:
            if image[j][i] == 1:
                # above
                if j != y[-1]:
                    if image[j+1][i] == 0:
                        px += [i, i+1]
                        py += [j+1, j+1]
                        ax.plot([i, i+1], [j+1, j+1], color="g", lw=2)
                # below
                if j != 0:
                    if image[j-1][i] == 0:
                        px += [i, i+1]
                        py += [j, j]
                        ax.plot([i, i+1], [j, j], color="g", lw=2)
                # left
                if i != 0:
                    if image[j][i-1] == 0:
                        px += [i, i]
                        py += [j, j+1]
                        ax.plot([i, i], [j, j+1], color="g", lw=2)
                # right
                if i != x[-1]:
                    if image[j][i+1] == 0:
                        px += [i+1, i+1]
                        py += [j, j+1]
                        ax.plot([i+1, i+1], [j, j+1], color="g", lw=2)


def get_centroid_fw(fluxes, box_edge, init=None):
    if (box_edge % 2) == 0:
        L = box_edge / 2
        box_edge += 1
    else:
        L = (box_edge - 1) / 2

    if init is None:
        init = np.asarray(np.asarray(np.shape(fluxes[0])) / 2.0, int)
    else:
        init = np.asarray(np.round(init), int)

    print init

    xc, yc = [], []
    for img in fluxes:
        sub_img = img[init[1] - L:init[1] + L + 1, init[0] - L:init[0] + L + 1]

        Isum = np.sum(sub_img, axis=0)
        Jsum = np.sum(sub_img, axis=1)

        xedge = np.arange(np.shape(img)[1])[init[0] - L:init[0] + L + 1]
        yedge = np.arange(np.shape(img)[0])[init[1] - L:init[1] + L + 1]

        Ibar = (1.0 / box_edge) * np.sum(Isum)
        Jbar = (1.0 / box_edge) * np.sum(Jsum)

        Ibar_diff = Isum - Ibar
        Jbar_diff = Jsum - Jbar

        xc_top = np.sum((Ibar_diff * xedge)[Ibar_diff > 0])
        xc_bot = np.sum(Ibar_diff[Ibar_diff > 0])

        yc_top = np.sum((Jbar_diff * yedge)[Jbar_diff > 0])
        yc_bot = np.sum(Jbar_diff[Jbar_diff > 0])

        xc.append(xc_top / xc_bot)
        yc.append(yc_top / yc_bot)

    print xc
    print yc

    xc, yc = np.asarray(xc)-np.nanmean(xc), np.asarray(yc)-np.nanmean(yc)

    return xc, yc


if __name__ == "__main__":

    # for aperture in [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]:

    # for aperture in [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    #                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #                  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]:

    for aperture in [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]:
        aperture = np.asarray(aperture)
        # aperture = np.ones(aperture.shape, bool)

        tpf = KeplerTargetPixelFile.from_archive(247887989)

        im = np.sum(tpf.flux, axis=0) / tpf.flux.shape[0]
        cmap = matplotlib.cm.get_cmap("gray")
        cmap.set_bad(color='red', alpha=0.3)
        plt.imshow(im, cmap=cmap, extent=[0, len(im[0]), 0, len(im)], origin=[0, 0],
                   vmin=np.nanmedian(im)*5., vmax=np.nanmax(im, axis=(1, 0)),
                   norm=LogNorm(np.nanmedian(im)*5., np.nanmax(im, axis=(1, 0)))
                   )

        plot_contours(aperture, plt.gca())
        # plt.colorbar()
        plt.tight_layout(0.)
        plt.show()

        t = tpf.time
        bkg = np.nanmedian(tpf.flux, axis=(2, 1))

        aperture_fluxes = []  # array of aperture pixel fluxes - background [3D]
        for i in range(len(t)):
            im = (tpf.flux[i] - bkg[i]) * aperture
            aperture_fluxes.append(im)
        aperture_fluxes = np.asarray(aperture_fluxes)
        f = np.nansum(aperture_fluxes, axis=(2, 1))

        x, y = get_centroid_fw(aperture_fluxes, box_edge=7)

        # plt.plot(t, x, ".")
        # plt.plot(t, y, ".")
        # plt.show()

        mask = np.isfinite(t * f * x * y)
        # mask &= (q == 0)    # nan and quality mask
        mask &= t > 2988.5
        # mask &= t < 3064.7
        # mask &= np.arange(t.size) > 197

        with open("pars.pkl", "rb") as pf:
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

        t = t[mask]
        f = f[mask]
        # e = e[mask]
        x = x[mask]
        y = y[mask]
        # q = q[mask]
        tmask = tmask[mask]
        m_all = m_all[mask]
        m_4 = m_4[mask]

        f /= np.median(f)

        splits = [2997., 3033.]     # campaign 13
        s1 = np.nanargmin(np.abs(t - 2988.5))
        s2 = np.nanargmin(np.abs(t - 3064.7))

        # break_points = np.asarray([s1] + list(np.arange(s1, s2, 197)[1:]) + [s2])
        # break_points = np.asarray(list(np.arange(s1, s2, 197)[1:]) + [s2])
        break_points = np.asarray(list(np.arange(s1, s2, 210)[1:]) + [s2])
        # break_points = np.arange(0, t.size, 197)[1:]
        # break_points = np.asarray(list(np.arange(s1, s2, 197)[1:]))

        # tr_is = [np.argmin(np.abs(t - pars[3].t0 - i*pars[3].per)) for i in range(3)]   # [742, 1918, 3046]
        # nl, nu = 32, 53
        # break_points = np.array([tr_is[0]-nl, tr_is[0]+nu, tr_is[1]-nl, tr_is[1]+nu, tr_is[2]-nl, tr_is[2]+nu])

        # break_points = np.array([np.argmin(np.abs(t - split)) for split in splits])
        # break_points = np.asarray([406, 790, 1173, 1598, 1986, 2405, 2818, 3240])

        # print [len(sec) for sec in np.split(t, break_points)]
        # print t.size - break_points[-1]

        # tsplit, fsplit = np.split(t, break_points), np.split(f, break_points)
        # for i in range(len(break_points)):
        #     plt.plot(tsplit[i], fsplit[i], ".")
        # plt.show()

        # plt.plot(t, f, ".")
        # plt.plot(t, x, ".")
        # plt.plot(t, y, ".")
        # _ = [plt.axvline(t[v], color="0.8") for v in break_points]
        # plt.show()

        # f = f - m_4 + 1.      # all transits removed

        fc, tr, s, tr1, tr2 = correct(t, f, x*4., y*4., break_points, tmask, niters=4,
                                        method="spline",
                                        # plot_windows=True,
                                        # plot_arcs=True,
                                        # restore_trend=True,
                                        time_knotspacing=3.,
                                        rot_knotspacing=0.5,
                                        arc_knotspacing=0.5
                                        )

        plt.plot(t, fc, "k.")
        for pl in range(4):
            per, t0 = pars[pl].per, pars[pl].t0
            while t0 < t.min():
                t0 += per
            i = 0
            while t0 < t.max():
                plt.plot([t0, t0], [0.997, 0.9974], lw=3, color=["r", "b", "g", "y"][pl],
                         label=["b", "c", "d", "e"][pl] if i == 0 else None)
                t0 += per
                i += 1
        plt.show()


        # fmask = (t < 3018.36) | (t > 3018.43)
        # fc, tr, s, tr1, tr2, t, f, e, tmask = \
        #     fc[fmask], tr[fmask], s[fmask], tr1[fmask], tr2[fmask], t[fmask], f[fmask], e[fmask], tmask[fmask]
        # m_4, m_all = m_4[fmask], m_all[fmask]

        # print sum(fc == 1.), sum(tr == 1.)

        # fc = fc + m_4 - 1.  # re-add the transits

        # for brk in np.array_split(break_points, 3):
        #     i1, i2 = brk
        #     plt.plot(t[i1:i2][tmask[i1:i2]], f[i1:i2][tmask[i1:i2]], "ko", ms=2)
        #     plt.plot(t[i1:i2][~tmask[i1:i2]], f[i1:i2][~tmask[i1:i2]], "ro", ms=2)
        #     plt.plot(t[i1:i2], tr[i1:i2])
        #     plt.show()

        # tr = tr + m_4 - 1.

        # _ = [plt.axvline(v, color="0.8", ls="--") for v in splits]
        # plt.plot(t[tmask], f[tmask], "ko", ms=2)
        # plt.plot(t[~tmask], f[~tmask], "ro", ms=2)
        # plt.plot(t, tr)
        # plt.show()

        plt.figure(figsize=(10, 6))
        ftr1 = f-tr2+1.
        ftr2 = f-tr1+1.
        off1 = 0.003
        off2 = 2. * off1
        plt.plot(t[tmask], ftr1[tmask] + off1, "ko", ms=3)
        plt.plot(t[tmask], tr1[tmask] + off1, lw=2)
        plt.plot(t[tmask], ftr2[tmask] + off2, "ko", ms=3)
        plt.plot(t[tmask], tr2[tmask] + off2, lw=1.1)
        plt.plot(t, fc, "ko", ms=3)

        for pl in range(4):
            per, t0 = pars[pl].per, pars[pl].t0
            while t0 < t.min():
                t0 += per
            i = 0
            while t0 < t.max():
                plt.plot([t0, t0], [0.997, 0.9974], lw=3, color=["r", "b", "g", "y"][pl],
                         label=["b", "c", "d", "e"][pl] if i == 0 else None)
                t0 += per
                i += 1

        # _ = [plt.axvline(t[v], color="b", alpha=0.8) for v in break_points]
        # _ = [plt.text(t[v], 0.998, v, color="k") for v in break_points]
        # _ = [plt.axvline(v, color="r", alpha=0.6, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
        # ax2 = plt.gca().twinx()
        # # ax2.plot(t, x, "r.")
        # # ax2.plot(t, y, "b.")
        # ax2.plot(t, s, "b.")
        plt.legend(fontsize=12, loc=4, prop={'weight': 'bold'})
        plt.xlabel("Time (BJD - 2454833)", fontsize=15)
        plt.ylabel("Normalised flux", fontsize=15)
        plt.tight_layout(0.)
        # plt.savefig("k2-133-detrend.pdf")
        plt.show()

        # _ = [plt.axvline(v, color="0.8", ls="--") for v in splits]
        # plt.plot(t, fc, "ko", ms=3, lw=1, ls="-")
        # _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
        # plt.show()

        res = fc - m_4
        # res = fc[m_4 == 1.]
        # res -= res.mean()

        # plt.plot(res, ".")
        # plt.show()

        cdpp = np.median([np.std(res[i:i+13]) for i in range(0, len(res), 13)])
        print("6.5-hr CDPP = {:.2f} ppm".format(cdpp*1e6))

        err = cdpp * np.ones(t.size, float)
        # err = np.sqrt(sum(res**2.) / res.size) * np.ones(t.size, float)
        # print "Chi=1 error = {:.0f} ppm".format(err[0]*1e6)

        # plt.figure(figsize=(12, 6))
        # plt.errorbar(t, res, err, color="k", marker="o", ms=3, lw=1, ls="-")
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # for i in range(3):
        #     ph, fp = phase_fold(np.array_split(t, 3)[i], np.array_split(fc, 3)[i] - np.array_split(m_all, 3)[i] + 1.,
        #                         pars[3].per, pars[3].t0)
        #     plt.errorbar(ph, fp, err[0]*np.ones(ph.size, float), lw=0., marker=".", ms=10, elinewidth=1.5, color="k",
        #                  zorder=2)
        #     plt.plot(ph, fp, ".", ms=5, lw=1., ls="-", zorder=3, label=i)
        # mt = np.linspace(pars[3].t0 - 0.01 * pars[3].per, pars[3].t0 + 0.01 * pars[3].per, 1000)
        # bat = batman.TransitModel(pars[3], mt, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        # m = bat.light_curve(pars[3])
        # plt.plot(np.linspace(0.49, 0.51, 1000), m, lw=4, alpha=0.7, color="grey", zorder=1)
        # plt.xlim(0., 1.)
        # plt.ylim(0.9985, 1.0005)
        # plt.legend()
        # plt.show()

        # plt.plot(t, fc-m_all+1., "r.", ms=4)
        # plt.plot(t, fc-m_4+1., "b.", ms=4)
        # _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3) * pars[3].per + pars[3].t0]
        # plt.show()

        # np.savetxt("lc-mySFF.dat", np.array([t, fc, err]).T)
        # np.savetxt("lc-01-mySFF.dat", np.array([t, fc-m_all+1., err]).T)
        # np.savetxt("lc-none-mySFF.dat", np.array([t, fc-m_4+1., err]).T)

        mtot = np.ones(t.size, float)
        for i in range(3):
            params = pars[i]
            bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
            mtot += (bat.light_curve(params) - 1.)
        phase = (t - pars[3].t0 + pars[3].per/2.) / pars[3].per % 1
        ctmask = (phase >= 0.495) & (phase <= 0.505)
        # plt.plot(t[ctmask], fc[ctmask]-mtot[ctmask]+1., "k.")
        # plt.plot(t[~ctmask], fc[~ctmask]-mtot[~ctmask]+1., "r.")
        # plt.show()

        # plt.plot(t, e/np.nanmedian(lc.PDCSAP_FLUX))
        # plt.axhline(err[0])
        # plt.show()
        # np.savetxt("lc-mySFF-cut-transits.dat", np.array([t[ctmask], fc[ctmask]-mtot[ctmask]+1., err[ctmask]]).T)

        # plt.plot(t, res, "k.")
        # _ = [plt.axhline(ev, alpha=0.6, color="b") for ev in np.arange(-5, 5)*err[0]]
        # plt.show()

        # sns.distplot(res, bins=150, fit=stats.norm, rug=True, kde=False, color="k")
        # plt.show()

        # post_mask = t < 3064.7
        # post_mask &= np.arange(t.size) > 197
        # post_mask &= (fc - m_4) < err*7.
        # print(sum(~((fc - m_4) < err * 7.)))
        # fc = fc[post_mask]
        # t = t[post_mask]

        fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(14, 3.5), sharey=True)
        for pl in range(4):
            others = range(4)
            others.remove(pl)
            mtot = np.ones(t.size, float)
            for i in others:
                params = pars[i]
                bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
                mtot += (bat.light_curve(params) - 1.)

            ph, fp = phase_fold(t, fc - mtot + 1., pars[pl].per, pars[pl].t0)

            # axes[x[0]][x[1]].plot(ph, fp, "k.", ms=8, zorder=2)
            axes[pl].errorbar(ph, fp, err[0], lw=0., marker=".", ms=8, elinewidth=1.5, color="k", zorder=2)
            axes[pl].plot(ph, fp, ".", ms=3, lw=0.9, ls="", zorder=3, color="b")

            mt = np.linspace(pars[pl].t0 - 0.05 * pars[pl].per, pars[pl].t0 + 0.05 * pars[pl].per, 10000)
            bat = batman.TransitModel(pars[pl], mt, supersample_factor=15, exp_time=29.4 / 60. / 24.)
            m = bat.light_curve(pars[pl])
            axes[pl].plot(np.linspace(0.45, 0.55, 10000), m, lw=4, alpha=0.7, color="grey", zorder=1)

            depth = (1. - batman.TransitModel(pars[pl], np.array([pars[pl].t0]), supersample_factor=15,
                                              exp_time=29.4 / 60. / 24.).light_curve(pars[pl]))[0] * 1e6

            mpl = batman.TransitModel(pars[pl], t, supersample_factor=15, exp_time=29.4 / 60. / 24.).light_curve(pars[pl])

            pho = [0.025, 0.015, 0.008, 0.004][pl]
            axes[pl].set_xlim(0.5-pho, 0.5+pho)
            axes[pl].set_ylim(0.9978, 1.0008)
            axes[pl].text(0.5-pho/1.2, 0.9981, ["b", "c", "d", "e"][pl], fontsize=20, fontweight="bold")
        fig.text(0.5, 0.0, 'Orbital phase', ha='center', va='bottom', fontsize=18)
        fig.text(0.0, 0.5, 'Normalised flux', ha='left', va='center', rotation='vertical', fontsize=18)
        plt.tight_layout(pad=1.3)
        plt.savefig("k2-133-all-transits.pdf")
        plt.show()

        # fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
        #     ph, fp = phase_fold(t, fc - mtot + 1., pars[pl].per, pars[pl].t0)
        #
        #     # axes[x[0]][x[1]].plot(ph, fp, "k.", ms=8, zorder=2)
        #     axes[x[0]][x[1]].errorbar(ph, fp, err, lw=0., marker=".", ms=8, elinewidth=1.5, color="k", zorder=2)
        #     axes[x[0]][x[1]].plot(ph, fp, ".", ms=4, lw=0.9, ls="-", zorder=3, color="b")
        #
        #     mt = np.linspace(pars[pl].t0 - 0.1 * pars[pl].per, pars[pl].t0 + 0.1 * pars[pl].per, 1000)
        #     bat = batman.TransitModel(pars[pl], mt, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        #     m = bat.light_curve(pars[pl])
        #     axes[x[0]][x[1]].plot(np.linspace(0.4, 0.6, 1000), m, lw=4, alpha=0.7, color="grey", zorder=1)
        #
        #     depth = (1. - batman.TransitModel(pars[pl], np.array([pars[pl].t0]), supersample_factor=15,
        #                                       exp_time=29.4 / 60. / 24.).light_curve(pars[pl]))[0] * 1e6
        #
        #     mpl = batman.TransitModel(pars[pl], t, supersample_factor=15, exp_time=29.4 / 60. / 24.).light_curve(pars[pl])
        #     print (1. - np.mean(fc[mpl != 1.])) / err[0] * np.sqrt(np.float(sum(mpl != 1.)))
        #
        #     pho = [0.025, 0.015, 0.015, 0.005][pl]
        #     axes[x[0]][x[1]].set_xlim(0.5-pho, 0.5+pho)
        #     axes[x[0]][x[1]].set_ylim(0.9978, 1.0008)
        # plt.legend()
        # plt.tight_layout(pad=0.)
        # plt.savefig("poly-no-mask.pdf")
        # plt.show()
