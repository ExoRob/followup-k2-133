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
# for mask in [[11, 701, 741, 812, 1043, 1383, 1384, 1385, 2460, 2884, 3004, 3011, 3092, 3174, 3223, 3506], [0, 1, 2, 11, 12, 13, 14, 20, 26, 43, 270, 272, 516, 539, 591, 778, 893, 1109, 1121, 1163, 1252, 1278, 1292, 1301, 1322, 1388, 1691, 1865, 1933, 1968, 2024, 2121, 2228, 2259, 2573, 2695, 2746, 2830, 2832, 3004, 3125, 3242, 3258, 3384, 3558]]:
#     t = np.delete(t, mask)
#     f = np.delete(f, mask)
#
# with open("LC_K2SFF.dat", "w") as lcf:
#     for i in range(len(t)):
#         lcf.write("{},{},{}\n".format(t[i], f[i], 1e-4))


# Everest
