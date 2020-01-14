from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


d = fits.open("ktwo247887989-c13_lpd-targ.fits")
# print d[1].data.columns
data = d[1].data              # TPF table data

stamp_dim = d[2].data.shape   # shape of postage stamp, e.g. (7, 8)

bkg = np.nanmedian(data.FLUX, axis=(2, 1))  # background value at each cadence [1D]

# plt.plot(data.TIME, bkg)
# plt.axvline(3018.39)
# plt.show()

# flux = np.nansum(data.FLUX, axis=(2,1)) - bkg*np.prod(stamp_dim)  # total summed flux at each cadence [1D]
# plt.plot(data.TIME, flux)
# plt.axvline(3018.39)
# plt.show()

mask = np.isfinite(bkg)
bkg = bkg[mask]

stamp = np.nansum(data.FLUX[mask], axis=0) - bkg.sum()    # time-summed image minus background [2D]

pxl_fluxes = [data.FLUX[:, np.unravel_index(i, stamp_dim)[0], np.unravel_index(i, stamp_dim)[1]][mask] - bkg
              for i in range(np.prod(stamp_dim))]

# for pf in pxl_fluxes:
#     plt.plot(data.TIME[mask], pf, marker=".", ms=5)
#     plt.axvline(3018.39, ls="--", color="k")
#     plt.show()

# to_del = []
# for i in range(len(bkg)):
#     if not np.isfinite(bkg[i]):
#         to_del.append(i)
#     if not np.isfinite(np.nansum(stamp)):
#         to_del.append(i)

stamp = stamp[::-1]

bd = np.asarray([data.FLUX[mask][j][::-1][4, 0] - bkg[j] for j in range(len(bkg))])
# tg = np.nansum(data.FLUX[mask], axis=(2,1)) - bkg*np.prod(stamp.shape)
#
# plt.plot(data.TIME[mask], bd)
# plt.axvline(3018.39)
# plt.show()

ti = np.argmin(np.abs(data.TIME[mask] - 3018.39))
print ti
i1, i2 = ti-8, ti+10
fig = plt.figure(figsize=(8, 8))
fig.set_tight_layout(True)
im = plt.imshow(data.FLUX[mask][i1][::-1], animated=True, cmap='gray',
                extent=[0, len(stamp[0]), len(stamp), 0], norm=LogNorm(10**1.3, 10**3.1))


def updatefig(j):
    global new_im
    new_im = data.FLUX[mask][j][::-1]
    print j, np.nanmax(new_im, axis=(1, 0))

    im.set_array(new_im)
    plt.title(int(j), color="r" if ((data.TIME[mask][j] > 3018.34) & (data.TIME[mask][j] < 3018.44)) else "k")
    return im,


anim = FuncAnimation(fig, updatefig, frames=range(i1, i2), interval=300)
anim.save('test.mp4')
plt.show()

# plt.imshow(data.FLUX[mask][10][::-1], cmap='gray', extent=[0, len(stamp[0]), len(stamp), 0])
# plt.show()

# plt.plot(np.nansum(data.FLUX, axis=(2, 1)) - bkg, ".")
# plt.show()

# fig, ax = plt.subplots(1, figsize=(8, 8))   # plot stamp + aperture chosen
# ax.imshow(stamp, cmap='gray', extent=[0, len(stamp[0]), len(stamp), 0])  # plot time-summed image
# fig.tight_layout()
# plt.show()
