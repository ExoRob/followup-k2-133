from lightkurve import KeplerLightCurveFile, KeplerLightCurve
import matplotlib.pyplot as plt
import numpy as np
from exoplanet import phase_fold
import dill
import batman
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
import seaborn as sns
sns.set()

lc = KeplerLightCurveFile.from_archive(247887989, verbose=False).PDCSAP_FLUX

lc = lc.remove_nans()   # remove nans

# for v in vars(lc):
#     print v

per = 26.586174433368857
t0 = 3004.8633837577186


lc_flat, trend = lc.flatten(window_length=401, return_trend=True)   # flatten
# plt.errorbar(lc.time, lc.flux, yerr=lc.flux_err, ls="", marker=".", alpha=0.7)
# plt.plot(trend.time, trend.flux, lw=3, alpha=0.7)
# plt.show()

lc_rm, mask = lc_flat.remove_outliers(return_mask=True)     # remove outliers
# plt.plot(lc.time[mask==0], lc.flux[mask==0], ".")
# plt.plot(lc.time[mask], lc.flux[mask], ".", c="r", ms=15)
# plt.show()

tmask = np.ones(lc_rm.time.size, bool)  # mask all transits before detrending
with open("pars.pkl", "rb") as pf:
    pars = dill.load(pf)
models = []
for params in pars:
    bat = batman.TransitModel(params, lc_rm.time, supersample_factor=15, exp_time=29.4/60./24.0)
    m = bat.light_curve(params)
    models.append(m)
    tmask &= (m == 1.)

# plt.plot(lc_rm.time[tmask], lc_rm.flux[tmask], ".")
# plt.plot(lc_rm.time[tmask==0], lc_rm.flux[tmask==0], ".", c="r")
# plt.show()

tmask[-1] = True
lc_mask = lc_rm[tmask]  # no transits

lc_cor_mask = lc_mask.correct(windows=20, bins=15)     # SFF correct non-transit LC
# plt.errorbar(lc_cor.time, lc_cor.flux, yerr=lc_cor.flux_err, ls="", marker=".")
# plt.show()

sff_trend = lc_mask.flux/lc_cor_mask.flux       # SFF trend
sff_int = interp1d(lc_mask.time, sff_trend)     # interpolate to LC with transits
# plt.plot(lc_mask.time, sff_trend, ".")
# plt.plot(lc_mask.time, sff_int(lc_mask.time))
# plt.show()

lc_cor = KeplerLightCurve(time=lc_rm.time, flux=lc_rm.flux/sff_int(lc_rm.time))     # SFF corrected full LC
plt.plot(lc_rm.time, lc_rm.flux, ".", alpha=0.7)
plt.plot(lc_rm.time, sff_int(lc_rm.time), alpha=0.8, lw=2)
_ = [plt.axvline(t0 + per*i, c="k", ls="--") for i in range(3)]
plt.show()

# print lc.cdpp(transit_duration=4)       # initial CDPP
# print lc_cor.cdpp(transit_duration=4)   # detrended CDPP

fnone = lc_cor.flux - np.sum(np.asarray(models), axis=0) + 3.   # flux with all transits removed
clip = sigma_clip(fnone, sigma_lower=6., sigma_upper=3.).mask   # non-outliers
clip &= tmask
sec_mask = []
clip = clip == 0
plt.plot(lc_cor.time[clip], lc_cor.flux[clip], ".")                                     # good points
# plt.plot(lc_cor.time[clip==0], lc_cor.flux[clip==0], ls="", marker="X", c="r", ms=5)    # bad points
plt.plot(lc_cor.time, np.sum(np.asarray(models), axis=0) - 3.)  # total model
_ = [plt.axvline(t0 + per*i, c="k", ls="--") for i in range(3)]
_ = [plt.plot(lc_cor.time[clip][i], lc_cor.flux[clip][i], ls="", marker="X", ms=5, c="r") for i in sec_mask]
plt.show()

lc_save = lc_cor[clip]
tsave = np.delete(lc_save.time, sec_mask)
fsave = np.delete(lc_save.flux, sec_mask)
np.savetxt("LC_LKSFF.dat", np.array([tsave, fsave, np.ones(tsave.size, float)*np.std(fsave)]).T, delimiter=",")

f01 = (lc_cor.flux - np.sum(np.asarray(models)[:3], axis=0) + 2.)[clip]
phase, phase_flux = phase_fold(lc_save.time, f01, per, t0)
plt.errorbar(phase, phase_flux, yerr=np.nanmean(lc_cor_mask.flux_err), ls="", marker=".", alpha=0.7)
# plt.xlim(0.495, 0.505)
plt.show()
