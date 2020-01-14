import numpy as np
import matplotlib.pyplot as plt
import dill
import batman
from exoplanet import phase_fold
from scipy import stats

log2pi = np.log(2. * np.pi)


def lnlike(model, flux, error):
    """ log likelihood function """

    # Bayes log-likelihood
    sigma_sq = error**2.
    chi_sq = np.sum((flux - model)**2. / sigma_sq)
    ln_err = np.sum(np.log(error))
    log_like_transit = -flux.size/2.*log2pi - ln_err - chi_sq/2.

    # SciPy log-likelihood
    # log_like_transit = stats.norm(loc=model, scale=error).logpdf(flux).sum()

    if np.isfinite(log_like_transit):
        return log_like_transit

    else:
        return -np.inf


t, f, e = np.loadtxt("lc-mySFF.dat", unpack=True)

with open("../transit/pars.pkl", "rb") as pf:
    pars = dill.load(pf)
models = []
for i, params in enumerate(pars):
    bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.)
    models.append(bat.light_curve(params))
models = np.asarray(models)

pl = 2
m3 = np.sum(models[:pl], axis=0) + np.sum(models[pl+1:], axis=0) - 2.

f01 = f - m3 + 1.
# np.random.shuffle(f01)

# ph, fp = phase_fold(t, f-m3+1., pars[3].per, pars[3].t0)
# _, mp = phase_fold(t, models[3], pars[3].per, pars[3].t0)
# ph, fp, mp = np.asarray(ph), np.asarray(fp), np.asarray(mp)
# msk = (ph > 0.495) & (ph < 0.505)
# print "lnlike cut transit = {}".format(lnlike(mp[msk], fp[msk], e[:sum(msk)]) / sum(msk))

print "lnlike transit model = {}".format(lnlike(models[pl], f01, e) / t.size)
print "lnlike white noise   = {}".format(lnlike(1., f01, e) / t.size)      # like a Gaussian loc=1, sig=e

# plt.plot(ph[msk], fp[msk], ".")
# plt.plot(ph[msk], mp[msk])
# plt.show()

# plt.plot(t, f01, ".")
# plt.plot(t, models[pl])
# plt.show()

# f = f - np.sum(models, axis=0) + 4.
# lltm, llwn = [], []
# for i in range(1000):
#     np.random.shuffle(f)
#
#     lltm.append(lnlike(models[pl], f, e) / t.size)
#     # llwn.append(lnlike(1., f, e) / t.size)
#
# plt.hist(lltm)
# plt.axvline(7.13989451028)
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
# ax1.hist(lltm)
# ax2.hist(llwn)
# plt.show()
