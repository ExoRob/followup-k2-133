import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from exoplanet.gp import terms, GP
import my_constants as myc
from astropy.io import fits
import pymc3 as pm
import theano.tensor as tt

lc = fits.open("ktwo247887989-c13_llc.fits")[1].data

t = lc.TIME
f = lc.PDCSAP_FLUX
e = lc.PDCSAP_FLUX_ERR
q = lc.SAP_QUALITY
x = lc.PSF_CENTR1 - np.nanmean(lc.PSF_CENTR1)
y = lc.PSF_CENTR2 - np.nanmean(lc.PSF_CENTR2)

mask = np.isfinite(t * f * e * x * y) & (q == 0)  # nan and quality mask
mask &= t > 2988.5                          # poor pointing
mask &= ((t < 3018.36) | (t > 3018.42))     # asteroid

t = np.ascontiguousarray(t[mask], dtype=np.float64)
f = np.ascontiguousarray(f[mask], dtype=np.float64)
e = np.ascontiguousarray(e[mask], dtype=np.float64)
x = np.ascontiguousarray(x[mask], dtype=np.float64)
y = np.ascontiguousarray(y[mask], dtype=np.float64)
q = np.ascontiguousarray(q[mask], dtype=np.float64)

mu = np.median(f)
f = (f / mu - 1.) * 1e3
e = e * 1e3 / mu

# plot raw
# plt.errorbar(t, f, e, marker=".", ls="")
# plt.show()

results = xo.estimators.lomb_scargle_estimator(t, f, max_peaks=1, min_period=5.0, max_period=100.0, samples_per_peak=50)
peak = results["peaks"][0]
freq, power = results["periodogram"]

# plot periodogram
# plt.plot(1./freq, power, "k")
# plt.axvline(peak["period"], color="k", lw=4, alpha=0.3)
# plt.yticks([])
# plt.xlabel("period")
# plt.ylabel("power")
# plt.show()

with pm.Model() as model:
    # The mean flux of the time series
    mean = pm.Normal("mean", mu=0.0, sd=10.0)

    # A jitter term describing excess white noise
    logs2 = pm.Normal("logs2", mu=2.*np.log(np.min(e)), sd=5.0)

    # The parameters of the RotationTerm kernel
    logamp = pm.Normal("logamp", mu=np.log(np.var(t)), sd=5.0)
    logperiod = pm.Normal("logperiod", mu=np.log(peak["period"]), sd=5.0)
    logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
    logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
    mix = pm.Uniform("mix", lower=0, upper=1.0)

    # Track the period as a deterministic
    period = pm.Deterministic("period", tt.exp(logperiod))

    # Set up the Gaussian Process model
    kernel = xo.gp.terms.RotationTerm(
        log_amp=logamp,
        period=period,
        log_Q0=logQ0,
        log_deltaQ=logdeltaQ,
        mix=mix
    )

    # (x, y) GPs
    logS1 = pm.Normal("logS1", mu=0.0, sd=15.0, testval=np.log(np.var(x)))
    logw1 = pm.Normal("logw1", mu=0.0, sd=15.0, testval=np.log(3.0))
    logS2 = pm.Normal("logS2", mu=0.0, sd=15.0, testval=np.log(np.var(y)))
    logw2 = pm.Normal("logw2", mu=0.0, sd=15.0, testval=np.log(3.0))
    logQ = pm.Normal("logQ", mu=0.0, sd=15.0, testval=0)

    kernel_x = terms.SHOTerm(log_S0=logS1, log_w0=logw1, Q=1.0/np.sqrt(2.))
    kernel_y = terms.SHOTerm(log_S0=logS2, log_w0=logw2, log_Q=logQ)

    gp = xo.gp.GP(kernel, t, e**2. + tt.exp(logs2), J=4)
    gp_x = xo.gp.GP(kernel_x, x, e**2.)
    gp_y = xo.gp.GP(kernel_y, y, e**2.)

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    pm.Potential("loglike", gp.log_likelihood(f - mean)
                 + gp_x.log_likelihood(f - mean)
                 + gp_y.log_likelihood(f - mean))

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict() + gp_x.predict() + gp_y.predict())

    # Optimize to find the maximum a posteriori parameters
    map_soln = pm.find_MAP(start=model.test_point)

plt.plot(t, f, "k", label="data")
plt.plot(t, map_soln["pred"], color="C1", label="model")
plt.xlim(t.min(), t.max())
plt.legend(fontsize=10)
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
plt.show()

sampler = xo.PyMC3Sampler()
with model:
    sampler.tune(tune=2000, start=map_soln, step_kwargs=dict(target_accept=0.9))
    trace = sampler.sample(draws=2000)

print pm.summary(trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs2", "mean"])

# transit priors
st_r, st_r_u = 0.455, 0.036
st_m, st_m_u = 0.461, 0.015
t0, t0_u = 3004.8659, 0.01
per, per_u = 26.5837, 0.005
rp, rp_u = 0.0372, 0.005
inc, inc_u = 88.8, 1.
ecc, ecc_u = 0., 0.8
w, w_u = 0., 360.
qld = [0.5079, 0.2239]



