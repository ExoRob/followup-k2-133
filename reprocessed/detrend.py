from __future__ import print_function
import numpy as np
import math as mt
from k2sc.ls import fasper
import matplotlib.pyplot as plt
from time import time as ttime
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from astropy.io import fits


def psearch(_time, _flux, min_p, max_p):
    freq, power, nout, jmax, prob = fasper(_time, _flux, 6, 0.5)
    period = 1 / freq
    m = (period > min_p) & (period < max_p)
    period, power = period[m], power[m]
    j = np.argmax(power)

    # plt.plot(period, power)
    # plt.show()

    expy = mt.exp(-power[j])
    effm = 2 * nout / 6
    fap = expy * effm

    if fap > 0.01:
        fap = 1.0 - (1.0 - expy) ** effm

    return period[j], fap


def open_k2sc(fname, k2id, do_plots=False):
    # d = fits.getdata(fname, 1)      # load detrended LC
    d = fits.open(fname)[1].data

    m = np.isfinite(d.flux) & np.isfinite(d.time) & (~(d.mflags & 2 ** 3).astype(np.bool))  # mask
    m &= ~binary_dilation((d.quality & 2 ** 20) != 0)

    time = d.time[m]
    flux = (d.flux[m] - d.trtime[m] + np.nanmedian(d.trtime[m]) - d.trposi[m] + np.nanmedian(d.trposi[m]))
    mflux = np.nanmedian(flux)
    flux /= mflux
    flux_e = d.error[m] / mflux
    x = d.x[m]
    y = d.y[m]

    f_raw = d.flux[m] / mflux
    f_t = (d.flux[m] - d.trposi[m] + np.nanmedian(d.trposi[m])) / mflux     # flux corrected for position only
    f_p = (d.flux[m] - d.trtime[m] + np.nanmedian(d.trtime[m])) / mflux     # flux corrected for time only
    m_t = (d.trtime[m]) / np.nanmedian(d.trtime[m])                         # model for time
    m_p = (d.trposi[m]) / np.nanmedian(d.trposi[m])                         # model for position

    m_tot = d.trposi[m] + d.trtime[m] - np.nanmedian(d.trposi[m])

    t = time #+ 2454833.0    # time (BJD)
    f = flux                # normalised flux
    e = flux_e              # normalised flux uncertainty
    tfs = time - time[0]

    if do_plots:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 8), sharex=True)
        ax1.plot(t, f_raw, ".", color='k')
        ax1.plot(t, m_tot/np.median(m_tot), 'r', lw=2)
        ax2.plot(t, flux, ".", color="k")
        fig.savefig("detrend_plots/{}_mtot.pdf".format(k2id))
        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(3, figsize=(15, 8), sharex=True)
        axes[0].plot(t, flux, '.', color='k', markersize=4)
        axes[1].plot(t, f_t, '.', color='k', markersize=4)
        axes[1].plot(t, m_t, 'r', lw=2)
        axes[2].plot(t, f_p, '.', color='k', markersize=4)
        axes[2].plot(t, m_p, 'r', lw=0.8)
        # ax.plot(t, f/mflux-0.06, '.', color='k', markersize=4)
        plt.subplots_adjust(bottom=0.)
        plt.tight_layout()
        fig.savefig("detrend_plots/{}_k2sc.pdf".format(k2id))
        plt.show()
        plt.close(fig)

    return t, f, e


def detrend_k2sc(t, f, x, y, c, do_plots=False, k2id=None, transit_mask=None, max_de=150, max_time=300., npop=100,
                 force_basic_kernal=False):
    if do_plots: assert k2id
    import k2sc.detrender
    from k2sc.kernels import BasicKernelEP, QuasiPeriodicKernelEP, QuasiPeriodicKernel
    from k2sc.utils import sigma_clip
    from k2sc.de import DiffEvol
    default_splits = {0:None, 1:[1979.5,2021], 2:[2102], 3:[2153,2189], 4:[2240,2273], 5:[2344], 6:[2388,2428],
                      7:[2517.5], 8:[2578.5,2598], 9:None, 102:[2777], 111:[2830], 112:[2862], 12:[2916.5, 2951],
                      13:[2997,3032], 14:[3086.5,3123.5], 15:None, 16:[3297,3331], 17:[3367,3401.5]}

    if transit_mask is None:
        transit_mask = np.ones(t.size, bool)
    splits = default_splits[c]
    detrender = k2sc.detrender.Detrender(flux=f, inputs=np.transpose([t, x, y]), splits=splits, kernel=BasicKernelEP(),
                                         tr_nrandom=400, tr_nblocks=6, tr_bspan=50, mask=transit_mask)

    ttrend, ptrend = detrender.predict(detrender.kernel.pv0 + 1e-5, components=True)
    cflux = f - ptrend + np.median(ptrend) - ttrend + np.median(ttrend)
    cflux /= np.nanmedian(cflux)

    omask = sigma_clip(cflux, max_iter=10, max_sigma=5) & transit_mask

    nflux = f - ptrend + np.nanmedian(ptrend)
    ntime = t - t.mean()
    pflux = np.poly1d(np.polyfit(ntime[omask], nflux[omask], 9))(ntime)

    period, fap = psearch(t[omask], (nflux - pflux)[omask], 0.05, 65.)

    is_periodic = False
    if fap < 1e-50:
        print("> Found periodicity of {:.1f} days".format(period))
        is_periodic = True
        ls_fap = fap
        ls_period = period

    if force_basic_kernal:
        is_periodic = False

    kernel = QuasiPeriodicKernelEP(period=ls_period) if is_periodic else BasicKernelEP()

    inputs = np.transpose([t, x, y])  # inputs into K2SC
    detrender = k2sc.detrender.Detrender(f, inputs, mask=omask, kernel=kernel, tr_nrandom=500, splits=splits,
                                         tr_nblocks=20, tr_bspan=50)
    de = DiffEvol(detrender.neglnposterior, kernel.bounds, npop)

    if isinstance(kernel, QuasiPeriodicKernel):
        de._population[:, 2] = np.clip(np.random.normal(kernel.period, 0.1 * kernel.period, size=de.n_pop), 0.05, 25.)

    pbar = tqdm(total=100., initial=0, desc="Maximum time left")
    tstart_de = ttime()
    pc_before = 0.
    for i, r in enumerate(de(max_de)):
        tcur_de = ttime() - tstart_de
        pc_done = round(max([float(i) / max_de, tcur_de / max_time]) * 100., 2)
        pbar.update(pc_done - pc_before)
        pc_before = pc_done
        # print '  DE iteration %3i -ln(L) %4.1f' % (i, de.minimum_value), int(tcur_de), pc_done
        # stops after 150 iterations or 300 seconds
        if ((de._fitness.ptp() < 3) or (tcur_de > max_time)) and (i > 2):
            break
    pbar.close()
    print('   DE finished in {} seconds after {} iterations.'.format(int(tcur_de), int(i)))
    # '\n   DE minimum found at: %s' % np.array_str(de.minimum_location, precision=3, max_line_width=250),
    # '\n   DE -ln(L) %4.1f' % de.minimum_value)

    print('   Starting local hyperparameter optimisation...')
    pv, warn = detrender.train(de.minimum_location)
    print('   Local minimum found.')  # at: %s' % np.array_str(pv, precision=3))

    tr_time, tr_position = detrender.predict(pv, components=True)

    flux_cor = f - tr_time + np.nanmedian(tr_time) - tr_position + np.nanmedian(tr_position)

    mflux = np.nanmedian(f)
    f_t = (f - tr_position + np.nanmedian(tr_position)) / mflux     # flux corrected for position only
    f_p = (f - tr_time + np.nanmedian(tr_time)) / mflux             # flux corrected for time only
    m_t = tr_time / np.nanmedian(tr_time)                # model for time
    m_p = tr_position / np.nanmedian(tr_position)        # model for position

    m_tot = tr_time + tr_position - np.median(tr_position)

    # fig = detrender.plot_t()
    # fig = detrender.plot_xy()
    # fig = detrender.plot_report(detrender.kernel.pv0+1e-5, 247887989)
    # plt.show()

    if do_plots:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 8), sharex=True)
        ax1.plot(t, f, ".", color='k')
        ax1.plot(t, m_tot, 'r', lw=2)
        ax2.plot(t, flux_cor, ".", color="k")
        _ = [plt.axvline(v, color="k", ls="--") for v in splits]
        fig.savefig("{}_mtot.pdf".format(k2id))
        plt.close(fig)

        fig, axes = plt.subplots(3, figsize=(15, 8), sharex=True)
        axes[0].plot(t, flux_cor / mflux, '.', color='k', markersize=4)
        axes[1].plot(t, f_t, '.', color='k', markersize=4)
        axes[1].plot(t, m_t, 'r', lw=2)
        axes[2].plot(t, f_p, '.', color='k', markersize=4)
        axes[2].plot(t, m_p, 'r', lw=0.8)
        _ = [plt.axvline(v, color="k", ls="--") for v in splits]
        # ax.plot(t, f/mflux-0.06, '.', color='k', markersize=4)
        plt.subplots_adjust(bottom=0.)
        plt.tight_layout()
        fig.savefig("{}_k2sc.pdf".format(k2id))
        plt.close(fig)

    return flux_cor/mflux, [tr_time, tr_position, m_tot, detrender]


def detrend_sff(t, f, x, y, c, flat_window=401, sff_window=1, plot=False, k2id=None):
    if plot: assert k2id
    from lightkurve import KeplerLightCurve, SFFCorrector
    lc = KeplerLightCurve(time=t, flux=f, centroid_col=x, centroid_row=y, campaign=c, mission="K2")
    # lc.plot()
    # plt.show()
    lc = lc.flatten(window_length=flat_window)
    # corr_lc = lc.correct(windows=sff_window)

    corrector = SFFCorrector()
    corr_lc = corrector.correct(time=t, flux=lc.flux, centroid_col=x, centroid_row=y, windows=sff_window, niters=1)
    f_cor = corr_lc.flux

    if plot:
        ax1 = corrector._plot_rotated_centroids()
        plt.savefig("plots/c{}/{}_SFF1.pdf".format(c, k2id))
        plt.close("all")
        ax2 = corrector._plot_normflux_arclength()
        plt.savefig("plots/c{}/{}_SFF2.pdf".format(c, k2id))
        plt.close("all")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,8), sharex=True)
        ax1.plot(t, f, ".")         # raw
        ax2.plot(t, f_cor, ".")     # SFF corrected
        plt.savefig("plots/c{}/{}_SFF.pdf".format(c, k2id))
        plt.close("all")

    # p, fp = phase_fold(lc.time, lc.flux, 3.35578, 0.)
    # plt.plot(p, fp, ".")
    # plt.show()

    # from lightkurve.correctors import SFFCorrector
    # corrector = SFFCorrector()
    # s1 = time < 3297.
    # s2 = (time >= 3297.) & (time < 3331.)
    # s3 = time >= 3331.
    # ts, fs = [], []
    # for s in [s1, s2, s3]:
    #     # windows = int((time[s][-1]-time[s][0])/6.)
    #     windows = 1
    #     print(windows)
    #     corr_lc = corrector.correct(time=time[s], flux=sap_flux[s], centroid_col=x[s], centroid_row=y[s], windows=windows, niters=1
    #                                 # polyorder=5, niters=3, bins=15, sigma_1=3., sigma_2=5., restore_trend=False
    #                                 )
    #
    #     ax1 = corrector._plot_rotated_centroids()
    #     ax2 = corrector._plot_normflux_arclength()
    #     # plt.show()
    #
    #     # ax = lc.fold(period=0.995).plot(color='C0', alpha=0.2, label='With Motion')
    #     # ax = corr_lc.fold(period=0.995).plot(color='C3', alpha=0.2, label='Motion Corrected')
    #     # plt.legend()
    #     # corr_lc.plot()
    #     # plt.show()
    #
    #     ts.append(corr_lc.time)
    #     fs.append(corr_lc.flux)
    #
    # plt.show()
    # for j in range(3):
    #     plt.plot(ts[j], fs[j], ".")
    #     # p, fp = phase_fold(ts[j], fs[j], 3.35578, 0.)
    #     # plt.plot(p, fp, ".")
    # plt.show()

    return f_cor
