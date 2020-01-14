# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import dill
import sys
sys.path.append("/Users/rwells/MCcubed/")
import MCcubed as mc3
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")


def plot_mine(res):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bns, rmss, rmsg, rmsse = [], [], [], []
    for bn in range(1, 500):
        cut = res.size % bn
        p2 = np.asarray(res)[:-cut] if (cut > 0) else np.asarray(res)
        rfbin = p2.reshape(p2.size / bn, bn).mean(axis=1)

        # print bn, p2.size/bn, np.std(rfbin) * 1e6

        bns.append(bn)
        rmss.append(np.std(rfbin) * 1e6)

        N = bn
        M = float(p2.size / bn)
        gauss = rmss[0] / np.sqrt(N) * np.sqrt(float(M) / float(M - 1)) if bn > 1 else rmss[0]
        gauss_err = gauss / np.sqrt(2. * M)
        rmsg.append(gauss)
        rmsse.append(gauss_err)

    # ax.plot(bns, rmss, color="k")
    markers, caps, bars = \
        ax.errorbar(bns, rmss, yerr=rmsse, color="k", lw=1., label="K2 data", zorder=2, elinewidth=2., ecolor="0.5")
    _ = [bar.set_alpha(0.4) for bar in bars]
    ax.plot(bns, rmsg, color="r", lw=2)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    plt.tight_layout(0.)
    plt.show()


def plot_mc3(res):
    rms, rmsl, rmsh, stderr, bins = mc3.rednoise.binrms(res, 500)
    # rms, rmsl, rmsh, stderr = rms*1e6, rmsl*1e6, rmsh*1e6, stderr*1e6
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    markers, caps, bars = \
        ax.errorbar(bins, rms, [rmsl, rmsh], color="k", lw=1., label="K2 data", zorder=2, elinewidth=1.5, ecolor="0.5")
    _ = [bar.set_alpha(0.7) for bar in bars]
    ax.plot(bins, stderr, "r-", label="Gaussian", lw=2., zorder=5)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel("Bin size", fontsize=15)
    ax.set_ylabel("RMS (ppm)", fontsize=15)
    plt.legend()
    # ax2 = ax.twinx()
    # ax2.plot(bins, rms / stderr)
    plt.tight_layout(0.)
    plt.show()


with open("rednoise.pkl", "rb") as pf:
    res, t, tf = dill.load(pf)

N = tf.size

# draw = np.random.randint(0, res.size, N)
# res = res[draw]

# res = np.random.normal(0., 10., N)
# res += np.random.normal(0., 3., N)
# res += np.sin(2./100.*np.pi/np.arange(N))*np.random.normal(2.0, 2.0, N)

# resf = np.zeros(N, float)
# mask = np.array([np.nanmin(np.abs(v - t)) < 1./24./3. for v in tf])     # times in final LC

# plt.plot(tf[mask], res[mask], "k.")
# plt.plot(tf[~mask], res[~mask], "r.")
# plt.show()

# res = res[mask]


orig_ind = np.array([np.nanargmin(np.abs(v - tf)) for v in t])

res_full = np.zeros(N, float)
res_full[orig_ind] = res

n_nulls = sum(res_full == 0.)
res_full[res_full == 0.] = res[np.random.randint(0, res.size, n_nulls)]

plot_mc3(res_full)
