import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as tic
# from astropy.io import fits
# import batman
# import urllib
# from tqdm import tqdm
# import os
from scipy.interpolate import interp1d
import altaipony.fakeflares as ffl
import bayesflare.models.model as m
import dill
import my_constants as myc
import corner
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")

# epics = """247831507
# 247839422
# 247845752
# 247850729
# 247853139
# 247857741
# 247858390
# 247858453
# 247859143
# 247862098
# 247862900
# 247864498
# 247870344
# 247870569
# 247871295
# 247871738
# 247874534
# 247876252
# 247880949
# 247882085
# 247883267
# 247894072
# 247895322
# 247896007
# 247907172
# 247908250
# 247909274
# 247916899
# 247917882
# 247918080
# 247919604
# 247921677
# 247922410
# 247923794
# 247926159
# 247930016
# 247932205
# 247933788
# 247935061
# 247935687
# 247935696
# 247939258
# 247941378
# 247941613
# 247941930
# 247947563
# 247950654
# 247952042
# 247958685
# 247960397
# 247960696
# 247963829
# 247965109
# 247967408
# 247968420
# 247968501
# 247969163
# 247969715
# 247970815
# 247972566
# 247973705
# 247974606
# 247976253
# 247979530
# 247980762
# 247983269
# 247986139
# 247988689
# 247989703
# 247989931
# 247991214
# 247991373
# 247992574
# 247992575
# 247993234
# 247993651
# 247997069
# 247998793
# 247999780
# 248005824
# 248006392
# 248006676
# 248007162
# 248007785
# 248010214
# 248014510
# 248015397
# 248017479
# 248018164
# 248021334
# 248021602""".split("\n")
#
# lc_c = fits.open("ktwo247887989-c13_llc.fits")[1].data
# t_c = lc_c.TIME
# f_c = lc_c.PDCSAP_FLUX
# e_c = lc_c.PDCSAP_FLUX_ERR
# q_c = lc_c.SAP_QUALITY
# mask_c = np.isfinite(f_c) & np.isfinite(t_c)
# f_n_c = f_c[mask_c] / np.nanmedian(f_c[mask_c])
# # plt.plot(t_c[mask_c], f_n_c, "k.")
#
# corr = []
# # for epic in tqdm(epics):
# for j in [21, 49]:
#     epic = epics[j]
#
#     url1 = "https://archive.stsci.edu/missions/k2/lightcurves/c13/"
#     filename = 'ktwo' + epic + '-c13_llc.fits'
#     fullfilename = 'channel50/' + filename
#
#     is_file = os.path.isfile(fullfilename)  # check if already downloaded
#     if not is_file:
#         url2 = epic[:4] + '00000/' + epic[4:6] + '000/'
#         url = url1 + url2 + filename
#
#         dl = urllib.URLopener()
#         dl.retrieve(url, fullfilename)
#
#     lc = fits.open(fullfilename)[1].data
#     t = lc.TIME
#     f = lc.PDCSAP_FLUX
#     e = lc.PDCSAP_FLUX_ERR
#     q = lc.SAP_QUALITY
#
#     mask = np.isfinite(f) & np.isfinite(t)
#     f_n = f[mask] / np.nanmedian(f[mask])
#     # plt.plot(t[mask], f_n, ".")
#
#     ts1, ts2 = [], []
#     for i in range(t[mask].size):
#         t2i = np.abs(t[mask][i] - t_c[mask_c]).argmin()
#         if np.abs(t[mask][i] - t_c[mask_c][t2i]) < (0.5 * 29.4 / 60. / 24.):
#             ts1.append(i)
#             ts2.append(t2i)
#
#     # c = np.correlate(f_n[ts1], f_n_c[ts2])
#     # corr.append(c[0])
#
#     f = f_n_c[ts2] / f_n[ts1]
#     corr.append(np.std(f))
#
#     plt.plot(t[mask][ts1], f, ".", lw=1, ls="-")
#
# # plt.plot(range(len(corr)), corr, color="k", marker="o")
# plt.show()

t, f, e = np.loadtxt("lc-none-mySFF.dat", unpack=True)

# flares = (f - 1.) > 2.*e[0]
# # contig = np.append(flares[1:] & flares[:-1], False)
#
# contig = np.zeros(flares.size, bool)
# for i in range(1, flares.size):
#     if flares[i-1] and flares[i]:
#         contig[i-1:i+1] = True
#
# plt.plot(t[contig], f[contig], "r.", ls="-")
# plt.plot(t[~contig], f[~contig], "k.")
# plt.show()


def chi_sq(o, c):
    return np.sum((o - c)**2.)


lum = 0.033 * myc.LS

mask = (t > 3018.2) & (t < 3018.6)
# mask = np.ones(t.size, bool)
ft, ff, fe = t[mask], f[mask], e[mask]

# in_flare = (ft > 3018.34) & (ft < 3018.44)
# nrg = np.trapz((ff[in_flare] - 1.)*lum, x=ft[in_flare]*24.*3600.)
#
# print "Total energy = {} erg".format(nrg)

ft = (ft - ft[0])*24.

tspace = ft[1]
ti = np.linspace(ft[0]-tspace/2., ft[-1]+tspace/2., ft.size*100)       # in hours

# fl_ss = m.Flare(ts=ti, t0=4.5, amp=0.012).model(pdict={'t0': 4.5, 'taugauss': 0.002, 'tauexp': 0.2, 'amp': 0.12})
# fl = fl_ss.reshape(ft.size, 100).sum(axis=1) / 100.
# plt.plot(ti, fl_ss)
# plt.plot(ft, fl)
# plt.show()


with open("flare-posprob.pkl", "rb") as pklf:
    pos_all, prob_all, _ = dill.load(pklf)

# pos = pos_all[np.argmax(prob_all)]
# pos = pos_all[-1]

# plt.plot(prob_all)
# plt.axhline(np.percentile(prob_all, 10.))
# plt.show()

mask = [(prob_all > np.percentile(prob_all, 20.)) & (pos_all[:, 2] < 0.05) & (pos_all[:, 6] < 0.05) &
        (pos_all[:, 3] > .1) & (pos_all[:, 7] > .1) & ((pos_all[:, 4] - pos_all[:, 0]) < 0.385) &
        ((pos_all[:, 4] - pos_all[:, 0]) > 0.01)]  # & (pos_all[:, 1] < pos_all[:, 5])]
print pos_all[mask]

plt.plot(pos_all[mask][:, 4] - pos_all[mask][:, 0])
plt.show()

pos = pos_all[mask][0]

for pos in pos_all[mask]:
    flares_ss = np.array([m.Flare(ts=ti, t0=pos[0 + i * 4], amp=pos[1 + i * 4]).
                         model(
        pdict={'t0': pos[0 + i * 4], 'taugauss': pos[2 + i * 4], 'tauexp': pos[3 + i * 4], 'amp': pos[1 + i * 4]})
                          for i in range(2)])

    flares = np.array([fl.reshape(ft.size, 100).sum(axis=1) / 100. for fl in flares_ss])

    plt.figure(figsize=(7, 6))
    plt.plot(ft, ff, ".", ms=12, zorder=8, lw=0., label="Data", markerfacecolor="b", markeredgecolor="k",
             markeredgewidth=2)
    plt.plot(ti, flares_ss[0] + 1., "k--", label="Flare 1", zorder=5)
    plt.plot(ti, flares_ss[1] + 1., "k-.", label="Flare 2", zorder=6)
    plt.plot(ft, np.sum(flares, axis=0) + 1., "r-", label="Binned model", lw=2, zorder=7)
    plt.legend()
    plt.xlabel("Time (hours)", fontsize=15)
    plt.ylabel("Normalised flux", fontsize=15)
    plt.tight_layout(0.)
    plt.show()

s = "\nPeak time = {:.2f} hours\nAmplitude = {:.4f}\nRise time = {:.2f} hours\nDecay time = {:.2f} hours" \
    "\nPeak time = {:.2f} hours\nAmplitude = {:.4f}\nRise time = {:.2f} hours\nDecay time = {:.2f} hours\n"
print s.format(*pos)

# fig = plt.figure(figsize=(14, 8))
# for i in range(len(pos_all[0])):
#     fig.add_subplot(2, 4, i+1)
#     plt.hist(pos_all[:, i][prob_all > np.percentile(prob_all, 10.)], bins=100)
# plt.show()

flares_ss = np.array([m.Flare(ts=ti, t0=pos[0+i*4], amp=pos[1+i*4]).
                      model(pdict={'t0': pos[0+i*4], 'taugauss': pos[2+i*4], 'tauexp': pos[3+i*4], 'amp': pos[1+i*4]})
                      for i in range(2)])

flares = np.array([fl.reshape(ft.size, 100).sum(axis=1) / 100. for fl in flares_ss])

ti -= 1.8
ft -= 1.8

plt.figure(figsize=(7, 6))
plt.plot(ft, ff, ".", ms=12, zorder=8, lw=0., label="Data", markerfacecolor="b", markeredgecolor="k", markeredgewidth=2)

plt.plot(ti, flares_ss[0]+1., "k--", label="Flare 1", zorder=5)
plt.plot(ti, flares_ss[1]+1., "k-.", label="Flare 2", zorder=6)
# plt.plot(ti, np.sum(flares_ss, axis=0)+1., "k-", label="Flare 1 & 2")
plt.plot(ft, np.sum(flares, axis=0)+1., "r-", label="Binned model", lw=2, zorder=7)

plt.legend()
plt.xlim(0., 5.4)
# plt.ylim(0.999, 1.01)
plt.gca().xaxis.set_major_locator(tic.MultipleLocator(base=1.))
plt.gca().xaxis.set_minor_locator(tic.MultipleLocator(base=0.5))
plt.gca().yaxis.set_major_locator(tic.MultipleLocator(base=0.005))
plt.gca().yaxis.set_minor_locator(tic.MultipleLocator(base=0.001))
plt.xlabel("Time (hours)", fontsize=15)
plt.ylabel("Normalised flux", fontsize=15)
plt.tight_layout(0.)
plt.savefig("flare.pdf")
plt.show()


# n_sp = 50000
# tpeak_sp = np.random.uniform(3.5, 5.5, n_sp)
# tmid_sp = np.random.uniform(3.5, 6.5, n_sp)
# ampl_sp = np.random.uniform(0.005, 0.05, n_sp)
# tgaus_sp = np.random.uniform(0.001, 0.2, n_sp)
# texp_sp = np.random.uniform(0.01, 0.8, n_sp)
#
# fls_ss = np.array([m.Flare(ts=ti, t0=tpeak_sp[i], amp=ampl_sp[i]).model(pdict={'t0': tmid_sp[i],
#                                                                                'taugauss': tgaus_sp[i],
#                                                                                'tauexp': texp_sp[i],
#                                                                                'amp': ampl_sp[i]})
#                   + 1. for i in range(n_sp)])
#
# fls = np.array([fl.reshape(ft.size, 100).sum(axis=1) / 100. for fl in fls_ss])
#
# chis = np.array([chi_sq(ff, fls[i]) for i in range(n_sp)])
#
# # plt.plot(ti, fls_ss[10])
# # plt.plot(ft, fls[10])
# # plt.plot(ft, ff, "ko")
# # plt.show()
#
# plt.plot(chis)
# plt.show()
#
# # n_sp = 10000
# # tpeak_sp = np.random.uniform(3., 6., n_sp)
# # dur_sp = np.random.uniform(0.02, 0.3, n_sp)
# # ampl_sp = np.random.uniform(0.01, 0.3, n_sp)
# #
# # fls = np.array([ffl.aflare(t=ft, tpeak=tpeak_sp[i], dur=dur_sp[i], ampl=ampl_sp[i], upsample=True, uptime=100) + 1.
# #                 for i in range(n_sp)])
# # chis = np.array([chi_sq(ff, fls[i]) for i in range(n_sp)])
# #
# # samples = np.array([ar[chis > 0.002] for ar in [tpeak_sp, dur_sp, ampl_sp]]).T
# # corner.corner(samples, labels=["tpeak", "dur", "amp"], bins=50, use_math_text=True, plot_contours=True,
# #               label_kwargs={"fontsize": 15}, hist2d_kwargs={"bins": 50})
# # plt.show()
#
# plt.hist(chis, bins=n_sp/100)
# plt.show()
#
# ind = np.argmin(chis)
# tpeak, tmid, ampl, tgaus, texp = tpeak_sp[ind], tmid_sp[ind], ampl_sp[ind], tgaus_sp[ind], texp_sp[ind]
# s = "\nPeak time = {:.2f} hours\nMid time = {:.2f} hours\nAmplitude = {:.2f}%\nRise time = {:.2f} hours\n" \
#     "Decay time = {:.2f} hours\n"
# print s.format(tpeak, tmid, ampl*100., tgaus, texp)
#
# # best_flare = ffl.aflare(t=ft, tpeak=tpeak, dur=dur, ampl=ampl, upsample=True, uptime=100) + 1.
# # best_flare_ss = ffl.aflare(t=ti, tpeak=tpeak, dur=dur, ampl=ampl) + 1.
# best_flare = fls[ind]
# best_flare_ss = m.Flare(ts=ti, t0=tpeak, amp=ampl).model(pdict={'t0': tmid, 'taugauss': tgaus, 'tauexp': texp,
#                                                          'amp': ampl}) + 1.
#
#
# plt.figure(figsize=(8, 6))
# plt.errorbar(ft, ff, fe, lw=1.5, marker=".", ms=12, elinewidth=1.5, color="k", zorder=2, label="Data")
# # plt.plot(ft, ff, ".", ms=12, zorder=2, color="k", ls="-", lw=1.5)
# plt.plot(ft, ff, ".", ms=6, zorder=3, color="b")
# # plt.plot(ft[in_flare], ff[in_flare], ".", ms=4, zorder=3, color="r")
# # plt.plot(ft, ff, lw=0.9, ls="-", zorder=1, color="k")
#
# # fi = interp1d(ft[in_flare], ff[in_flare])(ti)
# # plt.fill_between(ti, fi, np.ones(ti.size, float), fi >= 1., interpolate=True, color="0.5", alpha=0.5, hatch="\\")
#
# plt.plot(ft, best_flare, label="Flare")
# plt.plot(ti, best_flare_ss, alpha=0.4, lw=3, label="Flare-SS")
# plt.plot(ft, ff-best_flare+1., label="Residuals")
#
# # TODO
# # model = linear ~ minute | exponential decay ~ hour
# # can have multiple flares in region
# # bin/convolve to match data
# # should integrate to the observations?
#
# plt.xlim(0., 9.)
# plt.ylim(0.999, 1.01)
# plt.gca().xaxis.set_major_locator(tic.MultipleLocator(base=1.))
# plt.gca().xaxis.set_minor_locator(tic.MultipleLocator(base=0.5))
# plt.gca().yaxis.set_major_locator(tic.MultipleLocator(base=0.002))
# plt.gca().yaxis.set_minor_locator(tic.MultipleLocator(base=0.0004))
# plt.xlabel("Time from flare start (hours)", fontsize=15)
# plt.ylabel("Normalised flux", fontsize=15)
# plt.legend()
# plt.tight_layout(0.)
# plt.savefig("flare.pdf")
# plt.show()
