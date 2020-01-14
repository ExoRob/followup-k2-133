import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as tic
import numpy as np
from astropy.io import fits
import seaborn as sns
sns.set()
sns.set_color_codes()
sns.set_style("ticks")

pix_scale = 0.009942    # arcsec / pixel
psf_fwhm = 0.0504461    # arcsec

arcsec, dmag = np.loadtxt("247887989I.tbl", skiprows=7, unpack=True, delimiter=",")

d = fits.open("247887989I.fits")
im = d[0].data[::-1]
y_width, x_width = im.shape

# im = np.abs(im)
# plt.hist(im.flatten(), bins=np.linspace(-10., 10, 200))
# print np.percentile(im.flatten()[im.flatten() > -9e3], [5., 50., 95.])
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
for i in range(3):
    im_plot = im + 1.5     # [0., .5, 1.][i]
    # im = np.ma.masked_where(im < 0., im)
    im_plot[im_plot < 0.] = 0.01

    axes[i].imshow(im_plot,
                   extent=np.array([x_width/2., -x_width/2., -y_width/2., y_width/2.])*pix_scale,
                   cmap="gist_heat",
                   norm=colors.LogNorm(vmin=[0.05, 0.1, 0.5][i]),
                   # vmin=0.01, vmax=6900.,
                   aspect="equal")
    axes[i].set_xlim(3, -3)
    axes[i].set_ylim(-3, 3)
plt.show()

im_min, im_max, im_med = np.min(np.abs(im), axis=(1, 0)), np.max(im, axis=(1, 0)), np.median(im, axis=(1, 0))
print im_min, im_med, im_max

# cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True, reverse=True, start=.4, hue=1.)
cmap = "gist_heat"

im[im < 0.] = 0.01

plt.imshow(im, cmap=cmap, extent=np.array([x_width/2., -x_width/2., -y_width/2., y_width/2.])*pix_scale,
           norm=colors.LogNorm(vmin=0.05), aspect="equal")
plt.show()

fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
ax1.plot(arcsec, dmag, color="k", lw=2)
ax1.invert_yaxis()
ax1.set_xlim(0., 3.)
ax1.set_ylim(None, 0.)

ax2 = fig.add_axes([0.33, 0.35, 0.8, 0.55])   # left, bottom, width, height
ax2.imshow(im, cmap=cmap, extent=np.array([x_width/2., -x_width/2., -y_width/2., y_width/2.])*pix_scale,
           norm=colors.LogNorm(vmin=0.05), aspect="equal")
ax2.set_xlim(3., -3.)
ax2.set_ylim(-3., 3.)

ax1.xaxis.set_major_locator(tic.MultipleLocator(base=1))
ax1.xaxis.set_minor_locator(tic.MultipleLocator(base=.2))
ax1.yaxis.set_major_locator(tic.MultipleLocator(base=1))
ax1.yaxis.set_minor_locator(tic.MultipleLocator(base=.5))

ax2.xaxis.set_major_locator(tic.MultipleLocator(base=1))
ax2.xaxis.set_minor_locator(tic.MultipleLocator(base=.5))
ax2.yaxis.set_major_locator(tic.MultipleLocator(base=1))
ax2.yaxis.set_minor_locator(tic.MultipleLocator(base=.5))

ax1.set_xlabel(r"$\Delta$ radial distance (arcsec)", fontsize=15)
ax1.set_ylabel(r"$\Delta$ mag", fontsize=15)
ax2.set_xlabel(r"$\Delta \alpha$ (arcsec)")
ax2.set_ylabel(r"$\Delta \delta$ (arcsec)")

ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_tick_params(direction='in', which='both')
ax2.yaxis.set_tick_params(direction='in', which='both')
plt.setp(ax2.spines.values(), color="white")
for i in range(len(ax2.get_xticklines())):
    ax2.get_xticklines()[i].set_color('white')
    ax2.get_yticklines()[i].set_color('white')
for i in range(len(ax2.xaxis.get_minorticklines())):
    ax2.xaxis.get_minorticklines()[i].set_color('white')
    ax2.yaxis.get_minorticklines()[i].set_color('white')

fig.tight_layout(pad=0.)
# fig.savefig("combined-plot.pdf")
plt.show()
