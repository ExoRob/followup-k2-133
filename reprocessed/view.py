import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import batman, transits
import pickle as dill
# import detrend
from exoplanet import phase_fold
import seaborn as sns
sns.set()
sns.set_color_codes()

import sys
version = int(sys.version[0])

KEPLER_QUALITY_FLAGS = {
    "1": "Attitude tweak",
    "2": "Safe mode",
    "4": "Coarse point",
    "8": "Earth point",
    "16": "Zero crossing",
    "32": "Desaturation event",
    "64": "Argabrightening",
    "128": "Cosmic ray",
    "256": "Manual exclude",
    "1024": "Sudden sensitivity dropout",
    "2048": "Impulsive outlier",
    "4096": "Argabrightening",
    "8192": "Cosmic ray",
    "16384": "Detector anomaly",
    "32768": "No fine point",
    "65536": "No data",
    "131072": "Rolling band",
    "262144": "Rolling band",
    "524288": "Possible thruster firing",
    "1048576": "Thruster firing"
}


def dec2bit(dec):
    b = 0 if dec == 0 else np.log2(dec)

    return b


# tpf = lk.KeplerTargetPixelFile("ktwo247887989-c13_lpd-targ.fits")
# tpf.plot()
# plt.show()

lc = lk.KeplerLightCurveFile("ktwo247887989-c13_llc.fits").PDCSAP_FLUX

# print lc.__dict__.keys()

t = lc.time
f = lc.flux
e = lc.flux_err
x = lc.centroid_row
y = lc.centroid_col
q = lc.quality

# for bit in set(q):
#     # print bit, KEPLER_QUALITY_FLAGS[str(bit)] if str(bit) in KEPLER_QUALITY_FLAGS else bit, list(q).count(bit)
#     print bit, dec2bit(bit), list(q).count(bit)

mask = np.isfinite(t * f * e * x * y) & (q == 0)
# for bad_bit in [8192, 524288]:
#     mask &= (q != bad_bit)

inv = (mask == 0)
# print mask.sum(), inv.sum()

with open("../transit/pars.pkl", "rb") as pf:
    if version == 3:
        pars = dill.load(pf, encoding='latin1')
    else:
        pars = dill.load(pf)

models = []
m_all, m_4 = np.ones(t.size, float), np.ones(t.size, float)
phs, fps, mps, qps, ms = [], [], [], [], []
tmask = np.ones(t.size, bool)
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.)
    m = bat.light_curve(params)
    models.append(m)
    tmask &= (m == 1.)

    if i < 3:
        m_all += (m - 1.)
    m_4 += (m - 1.)

    ph, fp = phase_fold(t, f, params.per, params.t0)
    _, mp = phase_fold(t, m, params.per, params.t0)
    _, qp = phase_fold(t, q, params.per, params.t0)
    phs.append(np.asarray(ph))
    fps.append(np.asarray(fp))
    mps.append(np.asarray(mp))
    qps.append(np.asarray(qp))


# i = 3
# med = np.nanmedian(f)

plt.figure(figsize=(14, 6))
plt.plot(t[mask], f[mask], ".", color="k", ms=3)
plt.plot(t[inv], f[inv], "X", color="r")
# plt.plot(phs[i][qps[i] == 0], fps[i][qps[i] == 0]/med, ".", color="k", ms=4)
# plt.plot(phs[i], mps[i])

# for b in set(q):
#     if b != 0:
#         mask = np.isfinite(t * f * e * x * y) & (q == b)
#         plt.plot(t[mask], f[mask], ".", label=b)
#         # plt.plot(phs[i][qps[i] == b], fps[i][qps[i] == b]/med, ".", label=b)
plt.legend()
plt.show()


# find outliers
t_ok = t[mask]
f_ok = f[mask]
f_ok /= np.median(f_ok)
rm = medfilt(f_ok, 71)     # running median
omask = (f_ok - rm) < (np.std(f_ok - rm + 1.)*5.)   # outlier mask

# plt.plot(t_ok, f_ok - rm + 1., ".")
# plt.axhline(np.std(f_ok-rm+1.)*5. + 1.)
# plt.show()
# plt.plot(t_ok[omask], f_ok[omask], ".")
# plt.plot(t_ok[omask == 0], f_ok[omask == 0], "X", color="r")
# plt.show()

t = t[mask][omask]
f = f[mask][omask]
e = e[mask][omask]
x = x[mask][omask]
y = y[mask][omask]
q = q[mask][omask]
tmask = tmask[mask][omask]
m_all = m_all[mask][omask]
m_4 = m_4[mask][omask]

print pars[3].t0 - t[0]

# plt.plot(t, f, ".")
# plt.plot(t, x - x.mean(), ".", alpha=0.4)
# plt.plot(t, y - y.mean(), ".", alpha=0.4)
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
# plt.show()

# fcor, det_pars = detrend.detrend_k2sc(t, f, y, x, 13, transit_mask=tmask, do_plots=True, k2id="K2-133",
#                                       max_de=1500, max_time=180*60., npop=150, force_basic_kernal=True)
# with open("fcor-basic.pkl", "wb") as pf:
#     dill.dump([t, f, e, fcor, det_pars], pf)

if version == 2:
    with open("fcor.pkl", "rb") as pf:
        fcor, det_pars = dill.load(pf)[3:]

with open("k2sc-det.pkl", "wb") as pf:
    dill.dump([t, fcor, e], pf)

if version == 3:
    with open("fcor-only.pkl", "rb") as pf:
        fcor = dill.load(pf, encoding='latin1')


# plt.plot(t, f, "k.", label="raw")
# plt.plot(t, det_pars[2]*m_4, label="total model")
# _ = [plt.axvline(v, color="r", alpha=0.4, lw=2) for v in np.arange(0, 3)*pars[3].per + pars[3].t0]
# plt.legend()
# plt.show()

# m01 = models[3][mask][omask]
# tr_inds = [i[0] for i in np.argwhere(m01 < 1.)]
# xtra = 5
# trs = [range(740-xtra, 744+xtra), range(1914-xtra, 1918+xtra), range(3042-xtra, 3046+xtra)]

# for i, tr in enumerate(trs):
#     ax = np.arange(4 + 2*xtra)
#     out_ax = np.append(ax[:xtra], ax[-xtra:])
#     out_f = np.append(f[tr][:xtra], f[tr][-xtra:])
#
#     p = np.poly1d(np.polyfit(out_ax, out_f, 3))
#
#     # plt.plot(out_ax, out_f, "o")
#     # plt.plot(out_ax, p(out_ax))
#
#     # phase, _ = phase_fold(t[tr], f[tr], pars[3].per, pars[3].t0)
#     # plt.plot(phase, f[tr], "o", c=sns.color_palette()[i])
#     # plt.plot(phase, p(ax), c=sns.color_palette()[i])
#     # plt.show()
#
#     cor = f[tr] - p(ax) + np.median(p(ax))
#     cor /= np.median(cor)
#     phase, cor_flux = phase_fold(t[tr], cor, pars[3].per, pars[3].t0)
#     plt.plot(phase, cor_flux, ".")
#
# plt.show()

fcor /= np.median(fcor)
# err = np.std(fcor[tmask])
err = np.median([np.std(fcor[tmask][i:i+12]) for i in range(0, len(fcor[tmask]), 12)])

# t_cut, f_cut = np.array([]), np.array([])
# oot = 5
# for i in [740, 1914, 3042]:
#     inds = np.arange(i - oot, i + 5 + oot)
#     # print sum(m_all[inds] < 1.)
#     t_cut = np.append(t_cut, t[inds])
#     f_cut = np.append(f_cut, fcor[inds] - m_all[inds] + 1.)
# np.savetxt("cut-transits.dat", np.array([t_cut, f_cut, err*np.ones(t_cut.size, float)]).T)

# for m in models:
#     print (1. - min(m)) / err * np.sqrt(sum(m < 1.))

# plt.plot(t, fcor, ".")
# plt.show()

# plt.plot(t, fcor - m_all + 1., ".")
# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
# for i in range(3):
#     i1, i2 = i*t.size/3, (i+1)*t.size/3
#     foldt = ((t[i1:i2] - pars[3].t0 + pars[3].per/2.) / pars[3].per) % 1
#     foldf = fcor[i1:i2] - m_all[i1:i2] + 1.
#     foldt, foldf = zip(*sorted(zip(foldt, foldf)))
#     foldt, foldf = np.asarray(foldt), np.asarray(foldf)
#
#     axes[i].errorbar(foldt, foldf, err, lw=0.8, marker=".", ms=10, elinewidth=1.5)    # capsize=4, capthick=1.5)
#     for j in [-1., 0., 1.]:
#         axes[i].axvline(0.5 + j*1./24./pars[3].per, c="k", ls="--", lw=0.8, alpha=0.7)
#     # axes[i].plot(np.asarray(pm01), fm01, lw=5, alpha=0.4)
#     axes[i].set_xlim(0.495, 0.505)
#     axes[i].set_ylim(1.-10.*err, 1.+6.*err)
# plt.show()

# for i in range(4):
#     ph_det, fp_det = phase_fold(t, fcor, pars[i].per, pars[i].t0)
#
#     plt.plot(ph_det, fp_det, ".")
#     plt.plot(phs[i], mps[i])
#     plt.show()


f01 = fcor-m_all+1.
# isum = [0, 1, 2]
# msum = np.sum(np.asarray([models[i][mask][omask] for i in isum]), axis=0)
# f01 = fcor - msum + len(isum)

# np.savetxt("lc_none.dat", np.array([t, fcor-m_4, err*np.ones(t.size, float)]).T)
# np.savetxt("lc_f01.dat", np.array([t, fcor-m_all, err*np.ones(t.size, float)]).T)

# per_range = (1., (t[-1]-t[0])/2.)
# sde_trans, bper, t0, rprs, pr = transits.bls_search(t, f01-1., err*np.ones(t.size, float), plot=True, nbin=t.size,
#                                                     per_range=per_range, q_range=(1./24./per_range[1], 5./24.),
#                                                     nf=10000, mf=50)
# print sde_trans, bper


assert version == 3, "This can only be run with Python 3"
from astropy.stats import BoxLeastSquares
from astropy import units as u
t *= u.day
durations = np.linspace(1./24., 5./24., 100) * u.day
model = BoxLeastSquares(t, f01, err*np.ones(t.size, float))
results = model.autopower(durations, minimum_period=1.*u.day, minimum_n_transit=3)
# print(results)

power = results.power #- medfilt(results.power, 101)
index = np.argmax(power)
period = results.period[index]
t0 = results.transit_time[index]
duration = results.duration[index]

print("\ni   = {:6d}\nPer = {:6.3f}\nT0  = {:6.3f}\nDur = {:6.3f}\n".format(index, period, t0, duration))

# sts = model.compute_stats(period, duration, t0)
# _ = [print(x, sts[x]) for x in sts.keys()]
# print()
# _ = [print(x, results[x]) for x in results.keys()]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
ax1.axvline(period.value, alpha=0.4, lw=3)
for n in range(2, 10):
    ax1.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
    ax1.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
ax1.plot(results.period, power, "k", lw=0.6)
# ax1.plot(results.period, results.power - medfilt(results.power, 101), "g", lw=0.6)
ax1.set_xlim(results.period.min().value, results.period.max().value)
ax1.set_ylim(0.)
# ax1.set_xlabel("period [days]")
ax1.set_ylabel("log likelihood")
# plt.show()

ax2.axvline(period.value, alpha=0.4, lw=3)
for n in range(2, 10):
    ax2.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
    ax2.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
ax2.plot(results.period, results.depth_snr, "k", lw=0.6)
ax2.set_xlim(results.period.min().value, results.period.max().value)
ax2.set_ylim(0.)
ax2.set_xlabel("period [days]")
ax2.set_ylabel("Depth SNR")
plt.show()

ph_det, fp_det = phase_fold(t, f01, period, t0)
ph_det = np.asarray(ph_det)
fp_det = np.asarray(fp_det)
_, m_det = phase_fold(t, model.model(t, period, duration, t0), period, t0)
plt.errorbar(ph_det, fp_det, err, lw=0.8, marker=".", ms=10, elinewidth=1.5)
plt.plot(ph_det, m_det)
plt.show()


# ph_det, fp_det = phase_fold(t, f01, pars[3].per, pars[3].t0)
# ph_det = np.asarray(ph_det) * 24. * pars[3].per
# fp_det = np.asarray(fp_det)
# plt.errorbar(ph_det, fp_det, err, lw=0.8, marker=".", ms=10, elinewidth=1.5)
# plt.plot(phs[3] * 24. * pars[3].per, mps[3])
# plt.show()

# t_old, f01_old = np.loadtxt("../transit/01_lc.dat", delimiter=",", unpack=True)
#
# plt.plot(t, f01, ".")
# plt.plot(t_old, f01_old, ".")
# for i in range(4)[3:]:
#     nt = 0
#     tt = pars[i].t0
#     while tt < t[-1]:
#         nt += 1
#         tt += pars[i].per
#
#     _ = [plt.axvline(v, color=["grey", "b", "g", "r"][i], alpha=0.4, lw=2) for v in
#          np.arange(0, nt)*pars[i].per + pars[i].t0]
# plt.show()
