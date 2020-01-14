import numpy as np
import matplotlib.pyplot as plt
import batman
from scipy.ndimage import median_filter, gaussian_filter
import dill
import pandas
from tqdm import tqdm
import my_constants as myc
from matplotlib import gridspec
import matplotlib.ticker as tic
import seaborn as sns
sns.set()
sns.set_color_codes()
sns.set_style("ticks")

#
sde_lim = 8.
per_tol = 0.01
tzero = 2988.5416652958083

# load all the data from the injection tests
rec_ar, depth_ar, sde_ar, pars_ar, param_dists, dp_ar, ntrans_ar, phase_ar, impact_ar, vals_ar = \
    np.array([]), np.array([]), np.array([]), np.array([]), [[]]*6, np.array([], int), np.array([]), np.array([]), \
    np.array([]), np.array([])

for i in range(0, 8):
    with open("inj-det/inj-res{}.pkl".format(i), "rb") as pf:
        rec_ar_tmp, depth_ar_tmp, sde_ar_tmp, pars_ar_tmp, dp_ar_tmp, ntrans_ar_tmp, phase_ar_tmp, impact_ar_tmp, \
            vals_ar_tmp, param_dists_tmp = dill.load(pf)
    rec_ar = np.append(rec_ar, rec_ar_tmp)
    depth_ar = np.append(depth_ar, depth_ar_tmp)
    sde_ar = np.append(sde_ar, sde_ar_tmp)
    pars_ar = np.append(pars_ar, pars_ar_tmp)
    dp_ar = np.append(dp_ar, dp_ar_tmp)
    ntrans_ar = np.append(ntrans_ar, ntrans_ar_tmp)
    phase_ar = np.append(phase_ar, phase_ar_tmp)
    impact_ar = np.append(impact_ar, impact_ar_tmp)
    vals_ar = np.append(vals_ar, vals_ar_tmp)

    for j in range(6):
        param_dists[j] = list(param_dists[j]) + list(param_dists_tmp[j])

n_tests = rec_ar.size
# n_tests = 2000

per_in = np.asarray(param_dists[0])[:n_tests]
t0_in = np.asarray(param_dists[1])[:n_tests]
sde_ar = sde_ar[:n_tests]
depth_ar = depth_ar[:n_tests]

per_out, ph_out, t0_out, dur_in = [], [], [], []
for i in tqdm(range(n_tests)):
    t0, period = vals_ar[i]['t0'], vals_ar[i]['per']    # fitted params

    if period >= 1. and tzero <= t0 < tzero+100.:      # set T0 equal to first epoch
        while t0-period > tzero:
            t0 -= period

        t_ss = np.linspace(t0_in[i]-per_in[i]/2., t0_in[i]+per_in[i]/2., 10000)     # one orbit
        m_lc = batman.TransitModel(pars_ar[i], t_ss, supersample_factor=15, exp_time=29.4 / 60. / 24.).\
            light_curve(pars_ar[i])     # model for one orbit
        inds_in_transit = np.argwhere(m_lc != 1.)   # in-transit points
        dur = float(inds_in_transit[-1] - inds_in_transit[0]) * (t_ss[-1] - t_ss[0]) / 10000.

        # phase = (t0 - 2988.5416652958083) / period
        # phase = (t0 - 2988.5416652958083) / pers[i]
        # if phase >= 1.-tol:
        #     phase -= 1.
        # ph_out.append(phase)

    else:
        period, t0, dur = [np.nan]*3

    per_out.append(period)
    t0_out.append(t0)
    dur_in.append(dur)

# ph_out = np.asarray(ph_out) % 1.
per_out = np.asarray(per_out)
t0_out = np.asarray(t0_out)
dur_in = np.asarray(dur_in)

dif_per = np.abs(per_in - per_out) / per_in
# dif_ph = np.abs(phase_ar - ph_out)
dif_t0 = np.abs(t0_in - t0_out)

rec_in_dur = dif_t0 <= dur_in

# plt.plot(per_in[rec_in_dur], dif_t0[rec_in_dur], "b.")
# plt.plot(per_in[~rec_in_dur], dif_t0[~rec_in_dur], "r.")
# plt.axhline(0.)
# plt.axhline(29.4/60./24.)
# plt.show()

rec_ar = np.ones(n_tests, bool)     # recovered? True/False
rec_ar &= (sde_ar >= sde_lim)       # above SDE limit
rec_ar &= (dif_per <= per_tol)      # period within tolerance
# rec_ar &= rec_in_dur
# rec_ar &= (dif_t0 <= dur_in)      # epoch within transit duration
# rec_ar &= (dif_ph <= per_tol)     # phase within tolerance

# plt.plot(t0_in[rec_in_dur], t0_out[rec_in_dur], "b.")
# plt.plot(t0_in[~rec_in_dur], t0_out[~rec_in_dur], "r.")
# plt.plot(pers[rec_ar], rec_in_dur[rec_ar], ".")
# plt.plot(pers, dur_in, ".")
# plt.plot(pers, dif_ph, ".")
# plt.axhline(tol, color="k", ls="--")
# plt.ylim(0.)
# plt.axhline(0.)
# plt.show()

# plt.plot(per_in[rec_in_dur], per_out[rec_in_dur], "b.")
# plt.plot(per_in[~rec_in_dur], per_out[~rec_in_dur], "r.")
# plt.show()

# plt.plot(phase_ar[~rec_ar], ph_out[~rec_ar], "r.", ms=3)
# plt.plot(phase_ar[rec_ar], ph_out[rec_ar], "k.", ms=3)
# plt.plot([0, 1], [0, 1])
# plt.show()

# plt.plot(t0s[~rec_ar], dif_t0[~rec_ar]/pers[i], "r.", ms=3)
# plt.plot(t0s[rec_ar], dif_t0[rec_ar]/pers[i], "k.", ms=3)
# plt.ylim(0., 0.5)
# plt.show()

# plt.plot(pers[~rec_ar], per_out[~rec_ar], "r.", ms=3, alpha=0.6)
# plt.plot(pers[rec_ar], per_out[rec_ar], "k.", ms=3, alpha=0.6)
# plt.plot([0, pers.max()], [0, pers.max()])
# plt.xlim(0, pers.max())
# plt.ylim(0, pers.max())
# plt.show()


# mask = (1024. < depth_ar) & (depth_ar < 1064.) & (dp_ar < 12) & (dp_ar > 8)
# print sum(rec_ar[mask] == 1), len(rec_ar[mask])
# mask = np.ones(rec_ar.size, bool)
#
# depth_ar = depth_ar[mask]
# rec_ar = rec_ar[mask]
# param_dists = [np.asarray(dist)[mask] for dist in param_dists]
#
# n_rec = rec_ar.sum()
# n_not = rec_ar.size - n_rec
# depth_rec = depth_ar[rec_ar == 1]
# depth_not = depth_ar[rec_ar == 0]

# dists = [pd for pd in param_dists] + [ntrans_ar, phase_ar, impact_ar]
# fig = plt.figure(figsize=(14, 8))
# for i in range(9):
#     fig.add_subplot(3, 3, i+1)
#     dist = np.asarray(dists[i])
#     dist_rec = dist[rec_ar]     # recovered
#     dist_not = dist[~rec_ar]    # missed
#     bins = np.linspace(dist.min(), dist.max(), min([len(set(dist)), 30]))
#
#     plt.hist(dist, bins=bins, color="0.4")
#     plt.hist(dist_rec, bins=bins, color="g", alpha=0.5)
#     plt.hist(dist_not, bins=bins, color="r", alpha=0.5)
#     plt.title(["Per", "T0", "Rp", "Inc", "Depth",  "DP", "NT", "Phase", "b"][i])
# plt.tight_layout(0.)
# plt.show()

# # sns.regplot(depth_ar, sde_ar)
# plt.plot(depth_ar, sde_ar, ms=3, marker="o", ls="", alpha=0.4)
# plt.axvline(1044., color="g", linestyle="-", label="Candidate")
# plt.axhline(8., color="k", linestyle="--", label="SDE limit")
# plt.xlabel("Depth (ppm)")
# plt.ylabel("SDE")
# plt.legend()
# plt.tight_layout()
# plt.show()

# cut to only region like candidate
cand_depth, delta_depth = 1009., 20     # ppm
cand_dp, delta_dp = 12, 4

# n_rec_dep = sum((1044.-delta_depth <= depth_rec) & (depth_rec <= 1044.+delta_depth))
# n_not_dep = sum((1044.-delta_depth <= depth_not) & (depth_not <= 1044.+delta_depth))
#
# print("\n{}/{} ({:.2f}%) recovered, {}-{} ppm.".
#       format(n_rec_dep, n_not_dep+n_rec_dep, n_rec_dep/float(n_rec_dep+n_not_dep)*100.,
#              1044-delta_depth, 1044+delta_depth))
#
# bins = np.linspace(400., 1400., 80)
# rec, notrec = np.histogram(depth_rec, bins=bins)[0], np.histogram(depth_not, bins=bins)[0]
# frac = rec.astype(np.float) / (rec + notrec)
# binplt = [np.average([bins[i], bins[i+1]]) for i in range(bins.size-1)]
#
# plt.plot(binplt, frac)
# plt.axvline(1044., color="k", linestyle="--", label="Candidate")
# plt.axvspan(1044.-delta_depth, 1044.+delta_depth, color="grey", alpha=0.5)
# plt.xlabel("Depth (ppm)")
# plt.ylabel("Fraction recovered")
# plt.legend()
# plt.tight_layout()
# plt.show()


# # make the nice plot
fig = plt.figure(figsize=(11, 6))
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1])
ax1 = fig.add_subplot(spec[0, 0])   # plot
ax2 = fig.add_subplot(spec[0, 1])   # colour bar

ax1.plot(per_in[rec_ar], depth_ar[rec_ar], "b.", ms=.5, alpha=0.99, zorder=2)     # recovered
ax1.plot(per_in[~rec_ar], depth_ar[~rec_ar], "r.", ms=.5, alpha=0.99, zorder=2)   # missed

for i in range(4):
    dep, per = [741.0, 1103., 1936., 1009.][i], [3.0714, 4.8678, 11.0243, 26.5837][i]
    if dep < 1390.:
        ax1.plot(per, dep, marker="*", ls="", ms=20, color="k", zorder=4)
        ax1.plot(per, dep, marker="*", ls="", ms=10, color="g", zorder=5)
    else:
        ax1.arrow(per, 1340., 0., 50., color="k", width=.3, head_width=.8, head_length=20.,
                  length_includes_head=True, zorder=4)
        ax1.arrow(per, 1345., 0., 40., color="g", width=.14, head_width=.42, head_length=11.,
                  length_includes_head=True, zorder=5)
    ax1.text(per+1., min([dep, 1360.]), ["b", "c", "d", "e"][i],
             fontweight="bold", fontsize=18, ha="center", va="center")

# c = sns.diverging_palette(10, 240, as_cmap=True)    # red-blue colours
c = "coolwarm_r"

pbins = np.linspace(1., 35., 17)        # period bins
dbins = np.linspace(1400., 400., 15)    # depth bins

im = np.zeros((dbins.size-1, pbins.size-1), float)      # image to over-plot
x = np.array([np.mean([pbins[i], pbins[i+1]]) for i in range(pbins.size-1)])
y = np.array([np.mean([dbins[i], dbins[i+1]]) for i in range(dbins.size-1)])

for i in range(dbins.size - 1):         # each row (from top down)
    d1, d2 = dbins[i + 1], dbins[i]         # depth range of bin

    for j in range(pbins.size - 1):     # each column
        p1, p2 = pbins[j], pbins[j + 1]     # period range of bin

        rec = rec_ar[(per_in >= p1) & (per_in < p2) & (depth_ar >= d1) & (depth_ar < d2)]   # recovered array

        if rec.size == 0:   # if no data
            im[i, j] = np.nan
        else:
            frac = 1. if (sum(rec == 0) == 0) else float(sum(rec == 1)) / rec.size
            im[i, j] = frac

        # ax1.text((p1 + p2)/2., (d1 + d2)/2., "{:.2f}".format(im[i, j]), ha="center", va="center")

with open("for_contour.pkl", "wb") as pf:
    dill.dump([rec_ar, im, pbins, dbins, per_in, depth_ar], pf)

# img = ax1.imshow(im*100., extent=(pbins.min(), pbins.max(), dbins.min(), dbins.max()),
#                  aspect="auto", cmap=c, alpha=0.6, vmax=100., vmin=0.)    # vmin=np.nanmin(im, axis=(1, 0))
# cbar = fig.colorbar(img, cax=ax2, ticks=np.arange(0, 101, 10), use_gridspec=True, drawedges=False)

# cset = plt.contour(x, y, gaussian_filter(im, 0.7)*100, np.arange(0., 101., 10.),
#                    linewidths=2, antialiased=True, colors="k")
# plt.clabel(cset, inline=True, fmt='%d%%', fontsize=20, colors="k", fontweight="heavy")

cont = ax1.contourf(np.linspace(pbins[0], pbins[-1], pbins.size-1), np.linspace(dbins[0], dbins[-1], dbins.size-1),
                    gaussian_filter(im, (0., 1.))*100., np.linspace(0., 100., 11),
                    antialiased=True, cmap="coolwarm_r", alpha=1, zorder=1)
cbar = fig.colorbar(cont, cax=ax2, ticks=np.arange(0, 101, 10))

for c in cont.collections:
    c.set_edgecolor("face")
    c.set_rasterized(True)

# cbar.solids.set(alpha=0.6)
# cbar.solids.set_rasterized(True)

ax1.set_xlim(pbins[0], pbins[-1])
ax1.set_ylim(dbins[-1], dbins[0])
ax1.set_xlabel("Period (days)", fontsize=15)
ax1.set_ylabel("Depth (ppm)", fontsize=15)
ax1.xaxis.set_minor_locator(tic.MultipleLocator(base=1))
fig.tight_layout(pad=0.)
fig.savefig("injection-test.pdf")
plt.show()
