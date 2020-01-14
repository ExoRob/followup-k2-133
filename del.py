from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# d = fits.open("/Users/rwells/Downloads/ktwo212737443-c06_llc.fits")
# print d[1].data.columns


d = fits.open("/Users/rwells/Downloads/ktwo212737443-c06_lpd-targ.fits")
# print d[1].data.columns
data = d[1].data              # TPF table data

stamp_dim = d[2].data.shape   # shape of postage stamp, e.g. (7, 8)

bkg = np.nanmedian(data.FLUX, axis=(2, 1))  # background value at each cadence [1D]
# flux = np.nansum(data.FLUX, axis=(2,1)) - bkg*np.prod(stamp_dim)  # total summed flux at each cadence [1D]

mask = np.isfinite(bkg)
bkg = bkg[mask]

stamp = np.nansum(data.FLUX[mask], axis=0) - bkg.sum()    # time-summed image minus background [2D]

# to_del = []
# for i in range(len(bkg)):
#     if not np.isfinite(bkg[i]):
#         to_del.append(i)
#     if not np.isfinite(np.nansum(stamp)):
#         to_del.append(i)

stamp = stamp[::-1]

# plt.plot(np.nansum(data.FLUX, axis=(2, 1)) - bkg, ".")
# plt.show()

fig, ax = plt.subplots(1, figsize=(8, 8))   # plot stamp + aperture chosen
ax.imshow(stamp, cmap='gray', extent=[0, len(stamp[0]), len(stamp), 0])  # plot time-summed image
fig.tight_layout()
plt.show()
