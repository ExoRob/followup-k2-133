import dill
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import seaborn as sns
from scipy import stats
from exoplanet import phase_fold
import numpy as np
sns.set()
sns.set_style("ticks")
sns.set_palette("Dark2")

# white & red noise
sde_ar_w, per_ar_w = [], []
sde_ar_r, per_ar_r = [], []

for i in range(0, 6):
    with open("bootstrap4/bootstrap_res{}.pkl".format(i), "rb") as pf:
        cand_sde, sde_ar_tmp, per_ar_tmp = dill.load(pf)
    sde_ar_w = np.append(sde_ar_w, sde_ar_tmp)
    per_ar_w = np.append(per_ar_w, per_ar_tmp)

for i in range(0, 10):
    with open("bootstrap5/bootstrap_res{}.pkl".format(i), "rb") as pf:
        cand_sde, sde_ar_tmp, per_ar_tmp = dill.load(pf)
    sde_ar_r = np.append(sde_ar_r, sde_ar_tmp)
    per_ar_r = np.append(per_ar_r, per_ar_tmp)

cand_sde = 10.7492147102

print sde_ar_w.size, sde_ar_r.size

# t, f, e = np.loadtxt("lc-none-mySFF.dat", unpack=True, dtype=float)
# ind = np.argmax(sde_ar)
# p, fp = phase_fold(t, f, per_ar[ind], t[0])
# plt.plot(p, fp, ".")
# plt.show()

# fps = np.sum(sde_ar > cand_sde)
# fp8 = np.sum(sde_ar > 8.)
# print fps, "/", len(sde_ar)
# print "FPP (c) =", float(fps)/len(sde_ar)*100., "%"
# print fp8, "/", len(sde_ar)
# print "FPP (8) =", float(fp8)/len(sde_ar)*100., "%"

# pers = per_ar[sde_ar > cand_sde]

# sns.distplot(per_ar[sde_ar > 8.], kde=False, bins=200)
# plt.axvline(26.6, c="k", ls="--")
# plt.xlim(0)
# plt.xlabel("Period (d)")
# plt.show()

# fit = stats.exponnorm.fit(sde_ar)
# exp = stats.exponnorm(K=fit[0], loc=fit[1], scale=fit[2])
# print fit
# x = np.linspace(0., 12., 1000)
# f = exp.pdf(x)

# print "P(> cand) = {:.3f} %\nP(> 8) = {:.3f} %".format((1.-exp.cdf(cand_sde))*1e2, (1.-exp.cdf(8.))*1e2)

kde_kws = {"lw": 2, "shade": False}
hist_kws = {"alpha": 0.3}

sns.kdeplot(sde_ar_w, cumulative=True)
line = plt.gca().lines[0]
cdf_w_x = line.get_xdata()
cdf_w_y = line.get_ydata()
plt.clf()
sns.kdeplot(sde_ar_r, cumulative=True)
line = plt.gca().lines[0]
cdf_r_x = line.get_xdata()
cdf_r_y = line.get_ydata()
plt.clf()
plt.close("all")

print (1. - cdf_w_y[np.argmin(np.abs(cdf_w_x - 10.7))]) * 100.
print (1. - cdf_r_y[np.argmin(np.abs(cdf_r_x - 10.7))]) * 100.
# print (1. - cdf_w_y[np.argmin(np.abs(cdf_w_x - 8.))]) * 100.
# print (1. - cdf_r_y[np.argmin(np.abs(cdf_r_x - 8.))]) * 100.

plt.figure(figsize=(7, 5))
sns.distplot(sde_ar_w, bins=50, kde=True, norm_hist=True, color="b", label="White noise",
             hist_kws=hist_kws, kde_kws=kde_kws)
sns.distplot(sde_ar_r, bins=50, kde=True, norm_hist=True, color="r", label="Red noise",
             hist_kws=hist_kws, kde_kws=kde_kws)
# plt.plot(x, f, lw=2, c="b")

# plt.plot(cdf_w_x, 1.-cdf_w_y, lw=2, c="b")
# plt.plot(cdf_r_x, 1.-cdf_r_y, lw=2, c="r")

plt.axvline(cand_sde, c="k", ls="--")
plt.text(cand_sde-0.1, 0.3, "Candidate \nSDE = {:.1f}".format(cand_sde),
         rotation=0., fontweight="bold", fontsize=12, ha="right")
# plt.axvline(8., c="k", ls=":")

plt.xlim(0., 12.)
# plt.ylim(0., 0.6)
plt.gca().xaxis.set_major_locator(tic.MultipleLocator(base=2))
plt.gca().xaxis.set_minor_locator(tic.MultipleLocator(base=1))
plt.gca().yaxis.set_major_locator(tic.MultipleLocator(base=0.1))
plt.gca().yaxis.set_minor_locator(tic.MultipleLocator(base=0.05))

plt.xlabel("SDE", fontsize=15)
plt.ylabel("Normalised count", fontsize=15)
plt.legend()
plt.tight_layout(0.)
plt.savefig("bootstrap-SDE.pdf")
plt.show()

# plt.figure(figsize=(14, 7))
# sns.distplot(sde_ar, bins=200, kde=True, hist=True, label="Bootstrap")
# plt.axvline(cand_sde, color="k", linestyle="--", label="Candidate")
# plt.xlabel("SDE")
# plt.legend()
# plt.tight_layout()
# plt.show()
