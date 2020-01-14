import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import dill
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

with open("for_contour.pkl", "rb") as pf:
    rec_ar, im, pbins, dbins, per_in, depth_ar = dill.load(pf)

# plt.imshow(im)
# plt.show()

# print len(rec_ar)
# import sys; sys.exit()

x = np.array([np.mean([pbins[i], pbins[i+1]]) for i in range(pbins.size-1)])
y = np.array([np.mean([dbins[i], dbins[i+1]]) for i in range(dbins.size-1)])

X, Y = np.meshgrid(x, y, copy=False)
X = X.flatten()
Y = Y.flatten()
Z = im

# A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
# B = Z.flatten()

# for thing in [X, Y, Z, A, B]:
#     print thing.shape

# coeff, r, rank, s = np.linalg.lstsq(A, B)
# print coeff
# print r
# print rank
# print s

# Z = gaussian_filter(Z, 1.)

cmap = cm.get_cmap("coolwarm_r")
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = 0.6
my_cmap = ListedColormap(my_cmap)

fig = plt.figure(figsize=(11, 6))
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1])
ax1 = fig.add_subplot(spec[0, 0])   # plot
ax2 = fig.add_subplot(spec[0, 1])   # colour bar

ax1.plot(per_in[rec_ar], depth_ar[rec_ar], "b.", ms=0.5, alpha=0.99, zorder=2)     # recovered
ax1.plot(per_in[~rec_ar], depth_ar[~rec_ar], "r.", ms=0.5, alpha=0.99, zorder=2)   # missed

# cset = ax1.contour(x, y, gaussian_filter(Z, 0.6)*100., np.linspace(0., 100., 11), colors="k")
# ax1.clabel(cset, inline=True, fmt='%.0f', fontsize=20, colors="k")

# for i in range(dbins.size - 1):         # each row (from top down)
#     d1, d2 = dbins[i + 1], dbins[i]         # depth range of bin
#     for j in range(pbins.size - 1):     # each column
#         p1, p2 = pbins[j], pbins[j + 1]     # period range of bin
#         ax1.text((p1 + p2)/2., (d1 + d2)/2., "{:.2f}".format(im[i, j]), ha="center", va="center", zorder=5)

# n_conts = 3
# for i in range(n_conts):
#     cont = ax1.contourf(x, y, gaussian_filter(Z, 0.7)*100., np.linspace(0., 100., 11), antialiased=False,
#                         cmap="coolwarm_r", zorder=0, alpha=1)
#     # for c in cont.collections:
#     #     c.set_edgecolor("face")

cont = ax1.contourf(np.linspace(pbins[0], pbins[-1], pbins.size-1), np.linspace(dbins[0], dbins[-1], dbins.size-1),
                    gaussian_filter(Z, (.0, 1.)) * 100., np.linspace(0., 100., 11), antialiased=True,
                    cmap="coolwarm_r", zorder=0, alpha=1)
cbar = fig.colorbar(cont, cax=ax2, ticks=np.linspace(0., 100., 11))

# img = ax1.imshow(Z*100, cmap=my_cmap, alpha=1, vmin=0., vmax=100.,
#                  extent=(pbins.min(), pbins.max(), dbins.min(), dbins.max()), aspect="auto")
# cbar = fig.colorbar(img, cax=ax2, ticks=np.arange(0, 101, 10), use_gridspec=True, drawedges=False)

for c in cont.collections:
    c.set_edgecolor("face")
    c.set_rasterized(True)

# cbar.solids.set(alpha=0.6)
# cbar.solids.set_rasterized(True)
# cbar.solids.set_edgecolor("face")

fig.tight_layout(pad=0.)
# plt.axis("auto")
ax1.set_xlim(pbins[0], pbins[-1])
ax1.set_ylim(dbins[-1], dbins[0])
plt.savefig("test.pdf")
plt.show()
