import matplotlib
# matplotlib.use("agg")
from lightkurve import search_targetpixelfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
seaborn.set_style("ticks")
seaborn.set_color_codes()


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


t0, per = 3004.8649, 26.5848

tpf = search_targetpixelfile('247887989').download()

# tpf.plot(aperture_mask=tpf.pipeline_mask)
# plt.show()

# lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
# lc.remove_outliers().plot(normalize=False, alpha=0.7)
# plt.show()

aperture = tpf.flux[0] < 200
npix = len(np.where(aperture == True)[0])
# tpf.plot(aperture_mask=aperture)
# plt.show()

# bkg_lc = tpf.to_lightcurve(aperture_mask=aperture)
# bkg_lc /= npix
# bkg_lc.remove_outliers().plot(normalize=False, alpha=0.7)
# plt.show()

tpfs = search_targetpixelfile('247887989', radius=2000, limit=5).download_all()
print tpfs

for i in range(len(tpfs)):
    separation = np.sqrt((tpfs[0].column - tpfs[i].column)**2 + (tpfs[0].row - tpfs[i].row)**2)
    print tpfs.data[i]._hdu[0].header["KEPLERID"], \
        (tpfs[i].column, tpfs[i].row), tpfs.data[i]._hdu[0].header["KEPMAG"], \
        '{:.02} pixels apart'.format(separation)

kids = []
bkgs = []
# fig, ax = plt.subplots(figsize=(8, 5))
fig_st, axes_st = plt.subplots(1, 5, figsize=(12, 5))
for i, t in enumerate(tpfs.data):
    # Construct a background aperture
    aper = np.nan_to_num(t.flux[0]) < 200
    npix = len(np.where(aper == True)[0])

    # Create a lightcurve
    bkg_lc = t.to_lightcurve(aperture_mask=aper)
    bkg_lc /= npix  # Don't forget to normalize by the number of pixels in the aperture!

    aper_in = np.nan_to_num(t.flux[0]) > 200
    npix_in = len(np.where(aper_in == True)[0])
    bkg_lc_in = t.to_lightcurve(aperture_mask=aper_in)
    bkg_lc_in /= npix_in

    print np.mean(bkg_lc.flux) / np.mean(bkg_lc_in.flux) * 1e6

    im = np.sum(t.flux, axis=0) / t.flux.shape[0]
    cmap = matplotlib.cm.get_cmap("gray")
    cmap.set_bad(color='red', alpha=0.3)
    axes_st[i].imshow(im, cmap=cmap, extent=[0, len(im[0]), 0, len(im)], origin=[0, 0],
               # vmin=1e2, vmax=3e4,
               # norm=LogNorm(1e2, 3e4)
               )
    plot_contours(aper, axes_st[i])

    # bkg_lc.plot(ax=ax, normalize=True, label=t.targetid)

    kids.append(t._hdu[0].header["KEPLERID"])
    bkgs.append(bkg_lc.flux)

# _ = [plt.axvline(t0 + i*per, color="r", ls="--", alpha=0.7) for i in np.arange(3)]
plt.show()

plt.plot(bkg_lc.time, bkgs[0] / bkgs[1], label=kids[1])
plt.plot(bkg_lc.time, bkgs[0] / bkgs[2], label=kids[2])
plt.plot(bkg_lc.time, bkgs[0] / bkgs[3], label=kids[3])
plt.plot(bkg_lc.time, bkgs[0] / bkgs[4], label=kids[4])
_ = [plt.axvline(t0 + i*per, color="r", ls="--", alpha=0.7) for i in np.arange(3)]
plt.legend()
plt.show()
