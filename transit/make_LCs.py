from __future__ import print_function
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from PyAstronomy import pyasl
import numpy as np
from scipy import interpolate
from scipy.ndimage import binary_dilation


# K2SC (not transit-masked)
# d = fits.getdata("EPIC_247887989_mast.fits", 1)  # load detrended LC
# m = np.isfinite(d.flux) & np.isfinite(d.time) & (~(d.mflags & 2 ** 3).astype(np.bool))
# m &= ~binary_dilation((d.quality & 2 ** 20) != 0)
#
# t = d.time[m]
# f = (d.flux[m] - d.trtime[m] + np.nanmedian(d.trtime[m]) - d.trposi[m] + np.nanmedian(d.trposi[m]))
# mflux = np.nanmedian(f)
# f /= mflux
# e = d.error[m] / mflux
#
# for mask in [[134, 139, 465, 485, 490, 544, 545, 594, 693, 729, 869, 1250, 1323, 1591, 1593, 1743, 1821, 1840, 1856, 1912, 1941, 1943, 2009, 2102, 2128, 2499, 2651, 2758, 3121, 3260]]:
#     t = np.delete(t, mask)
#     f = np.delete(f, mask)
#     e = np.delete(e, mask)
#
# with open("LC_K2SC.dat", "w") as lcf:
#     for i in range(len(t)):
#         lcf.write("{},{},{}\n".format(t[i], f[i], e[i]))


# K2SC (transit-masked)
import dill, batman
with open("bat_pars.pkl", "rb") as pf:
    pars = dill.load(pf)

d = fits.getdata("EPIC_247887989_mast.fits", 1)  # load detrended LC
m = np.isfinite(d.flux) & np.isfinite(d.time) & (~(d.mflags & 2 ** 3).astype(np.bool)) & np.isfinite(d.x) & \
    np.isfinite(d.y)
m &= ~binary_dilation((d.quality & 2 ** 20) != 0)

t = d.time[m]
f = d.flux[m]
x = d.x[m]
y = d.y[m]

transit_mask = np.ones(t.size, bool)
models = []
for par in pars:
    bat = batman.TransitModel(par, t, supersample_factor=15, exp_time=29.4/60./24.0)
    model = bat.light_curve(par)
    models.append(model)
    transit_mask[model != 1.] = False
full_transit_model = np.sum(np.asarray(models), axis=0) - 3.

# t = t[transit_mask]
# f = f[transit_mask]
# x = x[transit_mask]
# y = y[transit_mask]

# plt.plot(t, f, ".")
# plt.plot(t, x, ".")
# plt.plot(t, y, ".")
# plt.show()

max_time, max_de = 3600., 10000
# max_time, max_de = 30., 10000

import math as mt
import k2sc.detrender
from k2sc.kernels import BasicKernelEP, QuasiPeriodicKernelEP, QuasiPeriodicKernel
from k2sc.ls import fasper
from k2sc.utils import sigma_clip
from k2sc.de import DiffEvol
from tqdm import tqdm
from time import time as ttime


def psearch(_time, _flux, min_p, max_p):
    freq, power, nout, jmax, prob = fasper(_time, _flux, 6, 0.5)
    period = 1 / freq
    m = (period > min_p) & (period < max_p)
    period, power = period[m], power[m]
    j = np.argmax(power)

    expy = mt.exp(-power[j])
    effm = 2 * nout / 6
    fap = expy * effm

    if fap > 0.01:
        fap = 1.0 - (1.0 - expy) ** effm

    return period[j], fap


detrender = k2sc.detrender.Detrender(flux=f, inputs=np.transpose([t, x, y]), splits=[2997,3033], kernel=BasicKernelEP(),
                                     tr_nrandom=400, tr_nblocks=6, tr_bspan=50, mask=transit_mask)

ttrend, ptrend = detrender.predict(detrender.kernel.pv0 + 1e-5, components=True)
cflux = f - ptrend + np.median(ptrend) - ttrend + np.median(ttrend)
cflux /= np.nanmedian(cflux)

# omask = sigma_clip(cflux, max_iter=10, max_sigma=5) & transit_mask
omask = transit_mask

nflux = f - ptrend + np.nanmedian(ptrend)
ntime = t - t.mean()
pflux = np.poly1d(np.polyfit(ntime[omask], nflux[omask], 9))(ntime)

period, fap = psearch(t[omask], (nflux - pflux)[omask], 0.05, 25.)

is_periodic = False
if fap < 1e-50:
    print("> Found periodicity of {:.1f} days".format(period))
    is_periodic = True
    ls_fap = fap
    ls_period = period

kernel = QuasiPeriodicKernelEP(period=ls_period) if is_periodic else BasicKernelEP()

inputs = np.transpose([t, x, y])  # inputs into K2SC
detrender = k2sc.detrender.Detrender(f, inputs, mask=omask, kernel=kernel, tr_nrandom=400, splits=[2997,3033],
                                     tr_nblocks=6, tr_bspan=50)
de = DiffEvol(detrender.neglnposterior, kernel.bounds, 100)

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
print('   DE finished in {} seconds.'.format(int(tcur_de)))#,
# '\n   DE minimum found at: %s' % np.array_str(de.minimum_location, precision=3, max_line_width=250),
# '\n   DE -ln(L) %4.1f' % de.minimum_value)

print('   Starting local hyperparameter optimisation...')
pv, warn = detrender.train(de.minimum_location)
print('   Local minimum found.') # at: %s' % np.array_str(pv, precision=3))

tr_time, tr_position = detrender.predict(pv, components=True)

flux_cor = f - tr_time + np.nanmedian(tr_time) - tr_position + np.nanmedian(tr_position)

mflux = np.nanmedian(flux_cor)
f_t = (f - tr_position + np.nanmedian(tr_position)) / mflux     # flux corrected for position only
f_p = (f - tr_time + np.nanmedian(tr_time)) / mflux             # flux corrected for time only
m_t = tr_time / np.nanmedian(tr_time)                # model for time
m_p = tr_position / np.nanmedian(tr_position)        # model for position

m_tot = tr_time + tr_position - np.median(tr_position)


fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 8), sharex=True)
ax1.plot(t, f/np.median(f), ".", color='k')
ax1.plot(t, m_tot/np.median(m_tot), 'r', lw=2)
ax2.plot(t, flux_cor/mflux, ".", color="k")
ax2.axhline(1., c="r", alpha=0.5)
fig.savefig("mtot.pdf")
plt.close(fig)

fig, axes = plt.subplots(3, figsize=(15, 8), sharex=True)
axes[0].plot(t, flux_cor / mflux, '.', color='k', markersize=4)
axes[1].plot(t, f_t, '.', color='k', markersize=4)
axes[1].plot(t, m_t, 'r', lw=2)
axes[2].plot(t, f_p, '.', color='k', markersize=4)
axes[2].plot(t, m_p, 'r', lw=0.8)
# ax.plot(t, f/mflux-0.06, '.', color='k', markersize=4)
plt.subplots_adjust(bottom=0.)
plt.tight_layout()
fig.savefig("k2sc.pdf")
plt.close(fig)

e = np.std(flux_cor[transit_mask]/mflux)

with open("LC_K2SC_mask_unclipped.dat", "w") as lcf:
    for i in range(len(t)):
        lcf.write("{},{},{}\n".format(t[i], flux_cor[i]/mflux, e))

with open("LC_K2SC_mask_unclipped_notransits.dat", "w") as lcf:
    for i in range(len(t)):
        lcf.write("{},{},{}\n".format(t[i], flux_cor[i]/mflux-full_transit_model[i]+1., e))


# t, f, e = np.loadtxt("LC_K2SC_mask.dat.bak", unpack=True, delimiter=",")
# for mask in [[25, 26, 282, 484, 485, 490, 544, 545, 568, 594, 693, 729, 1250, 1323, 1593, 1738, 1821, 1840, 1842, 1856, 1912, 1943, 2009, 2128, 2499, 2651, 2758, 3121, 3260]]:
#     t = np.delete(t, mask)
#     f = np.delete(f, mask)
# with open("LC_K2SC_mask.dat", "w") as lcf:
#     for i in range(t.size):
#         lcf.write("{},{},{}\n".format(t[i], f[i], np.std(f)))
# plt.plot(t, f, ".")
# plt.show()


# K2SFF
# d = fits.open("hlsp_k2sff_k2_lightcurve_247887989-c13_kepler_v1_llc.fits")
# # h = str(d[1].header)
# # print("\n".join([h[i:i+80] for i in range(0, len(h), 80)]))
# data = d[1].data    # T, FRAW, FCOR, ARCLENGTH, MOVING, CADENCENO
# t = data["T"]
# f = data["FCOR"]
# t_u = np.linspace(t[0], t[-1], int((t[-1]-t[0])/0.020432106))   # uniform spacing
# mf = medfilt(f, 25)     # median filter
# cv = interpolate.interp1d(t, mf)(t_u)   # interpolate trend to full LC
# gc = pyasl.broadGaussFast(t_u, cv, 0.05, edgeHandling="firstlast")  # gaussian convole to smooth
# gc = interpolate.interp1d(t_u, gc)(t)   # interpolate back to only data points from K2SFF
# f = f - gc + 1.    # correct LC
#
# import batman, dill
# with open("bat_pars.pkl", "rb") as pf:
#     pars = dill.load(pf)
# models = []
# for par in pars:
#     bat = batman.TransitModel(par, t, supersample_factor=15, exp_time=29.4 / 60. / 24.0)
#     m = bat.light_curve(par)
#     models.append(m)
# models = np.asarray(models)
# mtot = np.sum(models[:3], axis=0) - 2.
#
# # plt.plot(t, f, ".")
# # plt.plot(t, mtot)
# plt.plot(t, f-mtot+1., ".")
# plt.plot(t, models[3])
# plt.axhline(0.999, c="k", ls="--")
# plt.show()

# for mask in [[11, 701, 741, 812, 1043, 1383, 1384, 1385, 2460, 2884, 3004, 3011, 3092, 3174, 3223, 3506], [0, 1, 2, 11, 12, 13, 14, 20, 26, 43, 270, 272, 516, 539, 591, 778, 893, 1109, 1121, 1163, 1252, 1278, 1292, 1301, 1322, 1388, 1691, 1865, 1933, 1968, 2024, 2121, 2228, 2259, 2573, 2695, 2746, 2830, 2832, 3004, 3125, 3242, 3258, 3384, 3558]]:
#     t = np.delete(t, mask)
#     f = np.delete(f, mask)
#
# with open("LC_K2SFF.dat", "w") as lcf:
#     for i in range(len(t)):
#         lcf.write("{},{},{}\n".format(t[i], f[i], 1e-4))


# Everest
