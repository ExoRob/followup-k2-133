import numpy as np
import dill
import matplotlib.pyplot as plt
import batman
import sys
sys.path.append("/Users/rwells/MCcubed/")
import MCcubed as mc3
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")


# # test
# # Generate residuals signal:
# N = 1000
# # White-noise signal:
# white = np.random.normal(0, 5, N)
# # (Sinusoidal) time-correlated signal:
# red = np.sin(np.arange(N)/(0.1*N))*np.random.normal(1.0, 1.0, N)
#
# # Plot the time-correlated residuals signal:
# plt.figure(0)
# plt.clf()
# plt.plot(white+red, ".k")
# plt.ylabel("Residuals", fontsize=14)
#
# # Compute the residuals rms-vs-binsize:
# maxbins = N/5
# rms, rmslo, rmshi, stderr, binsz = mc3.rednoise.binrms(white+red, maxbins)
#
# # Plot the rms with error bars along with the Gaussian standard deviation curve:
# plt.figure(-6)
# plt.clf()
# plt.errorbar(binsz, rms, yerr=[rmslo, rmshi], fmt="k-", ecolor='0.5', capsize=0, label="Data RMS")
# plt.loglog(binsz, stderr, color='red', ls='-', lw=2, label="Gaussian std.")
# plt.xlim(1,200)
# plt.legend(loc="upper right")
# plt.xlabel("Bin size", fontsize=14)
# plt.ylabel("RMS", fontsize=14)
# plt.show()


t, f, e = np.loadtxt("final-lc-mySFF.dat", unpack=True)

with open("pars.pkl", "rb") as pf:
    pars = dill.load(pf)
m_all = np.ones(f.size, float)
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    m = bat.light_curve(params)
    m_all += (m - 1.)

res = f - m_all

rms, rmsl, rmsh, stderr, bins = mc3.rednoise.binrms(res, 1500)
rms, rmsl, rmsh, stderr = rms * 1e6, rmsl * 1e6, rmsh * 1e6, stderr * 1e6

print stderr

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
plt.xlim(1, 1500)
plt.ylim(6, 300)
plt.tight_layout(0.)
plt.savefig("red-noise-binning.pdf")
# plt.show()
plt.clf()
