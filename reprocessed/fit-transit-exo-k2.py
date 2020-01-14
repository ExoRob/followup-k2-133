import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
from astropy.stats import BoxLeastSquares
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import corner
import astropy.units as u
import dill
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# load the light curve
time, flux, err = np.loadtxt("final-lc-mySFF.dat", unpack=True)
flux -= 1.
flux *= 1e3

print(sum(~np.isfinite(time * flux * err)))

# TODO: cut transits

run_fit = True
n_pl_fit = 4

# set parameters
bls_p1, bls_p2 = 1, 50          # BLS period range
r_star_prior = 0.455, 0.022     # stellar radius, error
m_star_prior = 0.461, 0.011     # stellar mass, error
ecc_tv = 0.                    # eccentricity test value [avoid error for 258]

prior_t0 = np.array([2988.3168, 2990.7656, 2993.1739, 3004.8659])
width_t0 = np.array([0.01, 0.01, 0.01, 0.01])
prior_per = np.array([3.0712, 4.8682, 11.0234, 26.5837])
width_per = np.array([0.001, 0.001, 0.002, 0.005])
prior_rp = np.array([0.0255, 0.0288, 0.0393, 0.0372])
width_rp = np.array([0.005, 0.005, 0.005, 0.005])
prior_i = np.array([86., 87., 88., 88.8])
width_i = np.array([4., 3., 2., 1.])

# window = 25
# start = 50
# finish = 100
# tune = 500
# target_accept = 0.9
# chains = 2
# draws = 1000
window = 150
start = 300
finish = 500
tune = 4000
target_accept = 0.9
chains = 8
draws = 2500

# window = 100
# start = 200
# finish = 300
# tune = 3500
# target_accept = 0.9
# chains = 4
# draws = 2000

plot_bls = False
plot_lc_1 = True
plot_residuals = True
plot_lc_2 = True


# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
# BEGIN CODE

# run BLS search
from astropy.stats import BoxLeastSquares

m = np.zeros(len(time), dtype=bool)
period_grid = np.exp(np.linspace(np.log(1), np.log(30), 50000))
bls_results = []
periods = []
t0s = []
depths = []

# Compute the periodogram for each planet by iteratively masking out
# transits from the higher signal to noise planets. Here we're assuming
# that we know that there are exactly four planets.
for i in range(n_pl_fit):
    bls = BoxLeastSquares(time[~m], flux[~m])
    bls_power = bls.power(period_grid, np.arange(1.5, 3., 0.5)/24., oversample=10, objective="snr")
    bls_results.append(bls_power)

    # Save the highest peak as the planet candidate
    index = np.argmax(bls_power.power)
    periods.append(bls_power.period[index])
    t0s.append(bls_power.transit_time[index])
    depths.append(bls_power.depth[index])

    # Mask the data points that are in transit for this candidate
    m |= bls.transit_mask(time, periods[-1], 0.5, t0s[-1])

    oi = [2, 1, 0, 3][i]
    print(periods[i], t0s[i], prior_t0[oi] - time[0], prior_per[oi])

# Lets plot the initial transit estimates based on these periodograms:
if plot_bls:
    fig, axes = plt.subplots(4, 2, figsize=(15, 10))
    for i in range(n_pl_fit):
        # Plot the periodogram
        ax = axes[i, 0]
        ax.axvline(np.log10(periods[i]), color="C1", lw=5, alpha=0.8)
        ax.plot(np.log10(bls_results[i].period), bls_results[i].power, "k")
        ax.annotate("period = {0:.4f} d".format(periods[i]),
                    (0, 1), xycoords="axes fraction",
                    xytext=(5, -5), textcoords="offset points",
                    va="top", ha="left", fontsize=12)
        ax.set_ylabel("bls power")
        ax.set_yticks([])
        ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("log10(period)")

        # Plot the folded transit
        ax = axes[i, 1]
        p = periods[i]
        x_fold = (time - t0s[i] + 0.5*p) % p - 0.5*p
        m = np.abs(x_fold) < 0.4
        ax.plot(x_fold[m], flux[m], ".k")

        # Overplot the phase binned light curve
        bins = np.linspace(-0.41, 0.41, 32)
        denom, _ = np.histogram(x_fold, bins)
        num, _ = np.histogram(x_fold, bins, weights=flux)
        denom[num == 0] = 1.0
        ax.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1")

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylabel("relative flux [ppt]")
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time since transit")

    fig.subplots_adjust(hspace=0.02)
    plt.show()

# print(bls_t0)

if run_fit:
    # if plot_bls:
    #     fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    #
    #     # Plot the periodogram
    #     ax = axes[0]
    #     ax.axvline(np.log10(bls_period), color="C1", lw=5, alpha=0.8)
    #     ax.plot(np.log10(bls_power.period), bls_power.power, "k")
    #     ax.annotate("period = {0:.4f} d".format(bls_period),
    #                 (0, 1), xycoords="axes fraction",
    #                 xytext=(5, -5), textcoords="offset points",
    #                 va="top", ha="left", fontsize=12)
    #     ax.set_ylabel("bls power")
    #     ax.set_yticks([])
    #     ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
    #     ax.set_xlabel("log10(period)")
    #
    #     # Plot the folded transit
    #     ax = axes[1]
    #     x_fold = (time - bls_t0 + 0.5*bls_period) % bls_period - 0.5*bls_period
    #     m = np.abs(x_fold) < 0.4
    #     ax.plot(x_fold[m], flux[m], ".k")
    #
    #     # Overplot the phase binned light curve
    #     bins = np.linspace(-0.41, 0.41, 32)
    #     denom, _ = np.histogram(x_fold, bins)
    #     num, _ = np.histogram(x_fold, bins, weights=flux)
    #     denom[num == 0] = 1.0
    #     ax.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1")
    #
    #     ax.set_xlim(-0.3, 0.3)
    #     ax.set_ylabel("de-trended flux [ppt]")
    #     ax.set_xlabel("time since transit")
    #     plt.show()


    # To confirm that we didn't overfit the transit, we can look at the folded light curve for the PLD model near transit.
    # This shouldn't have any residual transit signal, and that looks correct here:
    # plt.figure(figsize=(10, 5))
    #
    # x_fold = (time - bls_t0 + 0.5*bls_period) % bls_period - 0.5*bls_period
    # m = np.abs(x_fold) < 0.3
    # plt.plot(x_fold[m], flux[m], ".k", ms=4)
    #
    # bins = np.linspace(-0.5, 0.5, 60)
    # denom, _ = np.histogram(x_fold, bins)
    # num, _ = np.histogram(x_fold, bins, weights=flux)
    # denom[num == 0] = 1.0
    # plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1", lw=2)
    # plt.xlim(-0.2, 0.2)
    # plt.xlabel("time since transit")
    # plt.ylabel("PLD model flux")
    # plt.show()


    # ## The transit model in PyMC3
    #
    # The transit model, initialization, and sampling are all nearly the same as the one in :ref:`together`, but we'll
    # use a [more informative prior on eccentricity](https://arxiv.org/abs/1306.4982).

    def build_model(mask=None, start=None):
        if mask is None:
            mask = np.ones(len(time), dtype=bool)

        with pm.Model() as model:
            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)           # LC mean (set to 0)
            u_star = xo.distributions.QuadLimbDark("u_star")    # quadratic limb-darkening

            # Stellar parameters
            r_star = pm.Normal("r_star", mu=r_star_prior[0], sd=r_star_prior[1])    # stellar radius
            m_star = pm.Normal("m_star", mu=m_star_prior[0], sd=m_star_prior[1])    # stellar mass

            # Prior to require physical parameters
            pm.Potential("r_star_prior", tt.switch(r_star > 0, 0, -np.inf))     # stellar radius > 0
            pm.Potential("m_star_prior", tt.switch(m_star > 0, 0, -np.inf))     # stellar mass > 0

            # Orbital parameters for the planets
            logP = pm.Normal("logP", mu=np.log(periods), sd=1., shape=n_pl_fit)        # ln(period)
            t0 = pm.Normal("t0", mu=np.array(t0s), sd=1., shape=n_pl_fit)                       # T0 [phase, 0-P]
            b = pm.Uniform("b", lower=0, upper=1., testval=0.5+np.zeros(n_pl_fit), shape=n_pl_fit)        # impact parameter
            logr = pm.Normal("logr", sd=1.0, shape=n_pl_fit, mu=0.5*np.log(1e-3*np.array(depths)) + np.log(r_star_prior[0]))     # ln(Rp)
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))       # Rp
            # ror = pm.Deterministic("ror", r_pl / r_star)        # Rp/Rs

            # This is the eccentricity prior from Kipping (2013) - https://arxiv.org/abs/1306.4982
            # ecc = pm.Beta("ecc", alpha=0.867, beta=3.03, testval=ecc_tv+np.zeros(n_pl_fit), shape=n_pl_fit)   # eccentricity
            # omega = xo.distributions.Angle("omega", shape=n_pl_fit)                     # omega
            ecc = np.zeros(n_pl_fit)
            omega = np.zeros(n_pl_fit)

            # Transit jitter & GP parameters
            logs2 = pm.Normal("logs2", mu=np.log(np.var(flux[mask])), sd=10)
            logw0_guess = np.log(2*np.pi/10)
            logw0 = pm.Normal("logw0", mu=logw0_guess, sd=10)

            # We'll parameterize using the maximum power (S_0 * w_0^4) instead of S_0 directly because this removes
            # some of the degeneracies between S_0 and omega_0
            logpower = pm.Normal("logpower", mu=np.log(np.var(flux[mask]))+4*logw0_guess, sd=10)
            logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))   # period

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

            # Compute the model light curve using starry
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(         # compute light curve
                orbit=orbit, r=r_pl, t=time[mask], texp=29.4/60./24.)*1e3
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)

            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
            gp = xo.gp.GP(kernel, time[mask], tt.exp(logs2) + tt.zeros(mask.sum()), J=2)
            pm.Potential("transit_obs", gp.log_likelihood(flux[mask] - light_curve))
            pm.Deterministic("gp_pred", gp.predict())

            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            map_soln = xo.optimize(start=start, vars=[logs2, logpower, logw0])
            map_soln = xo.optimize(start=map_soln, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
            map_soln = xo.optimize(start=map_soln, vars=[u_star])
            map_soln = xo.optimize(start=map_soln, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            # map_soln = xo.optimize(start=map_soln, vars=[ecc, omega])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower, logw0])
            map_soln = xo.optimize(start=map_soln)

        return model, map_soln


    model0, map_soln0 = build_model()


    def plot_light_curve(soln, mask=None):
        if mask is None:
            mask = np.ones(len(time), dtype=bool)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        ax = axes[0]
        ax.plot(time[mask], flux[mask], "k", label="data")
        gp_mod = soln["gp_pred"] + soln["mean"]
        ax.plot(time[mask], gp_mod, color="C2", label="gp model")
        ax.legend(fontsize=10)
        ax.set_ylabel("relative flux [ppt]")

        ax = axes[1]
        ax.plot(time[mask], flux[mask] - gp_mod, "k", label="de-trended data")
        for i, l in enumerate("bcde"[:n_pl_fit]):
            mod = soln["light_curves"][:, i]
            ax.plot(time[mask], mod, label="planet {0}".format(l))
        ax.legend(fontsize=10, loc=3)
        ax.set_ylabel("de-trended flux [ppt]")

        ax = axes[2]
        mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
        ax.plot(time[mask], flux[mask] - mod, "k")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("residuals [ppt]")
        ax.set_xlim(time[mask].min(), time[mask].max())
        ax.set_xlabel("time [days]")

        return fig


    # plot the initial light curve model:
    if plot_lc_1:
        plot_light_curve(map_soln0)
        plt.show()

    # do sigma clipping to remove significant outliers
    mod = map_soln0["gp_pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves"], axis=-1)
    resid = flux - mod
    rms = np.sqrt(np.median(resid**2))
    mask = np.abs(resid) < 5 * rms

    if plot_residuals:
        plt.figure(figsize=(10, 5))
        plt.plot(time, resid, "k", label="data")
        plt.plot(time[~mask], resid[~mask], "xr", label="outliers")
        plt.axhline(0, color="#aaaaaa", lw=1)
        plt.ylabel("residuals [ppt]")
        plt.xlabel("time [days]")
        plt.legend(fontsize=12, loc=3)
        plt.xlim(time.min(), time.max())
        plt.show()

    # re-build the model using the data without outliers.
    model, map_soln = build_model(mask, map_soln0)

    # plot with outliers removed
    if plot_lc_2:
        plot_light_curve(map_soln, mask)
        plt.show()

    # sample the model
    np.random.seed()
    sampler = xo.PyMC3Sampler(window=window, start=start, finish=finish)
    with model:
        burnin = sampler.tune(tune=tune, start=map_soln,
                              step_kwargs=dict(target_accept=target_accept),
                              chains=chains)
    with open("sampler.pkl", "wb") as pf:
        dill.dump([burnin], pf)

    with model:
        trace = sampler.sample(draws=draws, chains=chains)
    with open("sampler.pkl", "wb") as pf:
        dill.dump([burnin, trace], pf)

else:
    with open("sampler.pkl", "rb") as pf:
        burnin, trace = dill.load(pf)
    mask = np.ones(time.size, bool)

print(pm.summary(trace, varnames=["logw0", "logpower", "logs2", "r_pl", "b", "t0", "logP", "r_star",
                                  "m_star", "u_star", "mean"]))


# ## Results
# Compute the GP prediction
gp_mod = np.median(trace["gp_pred"] + trace["mean"][:, None], axis=0)

# Get the posterior median orbital parameters
p = np.median(trace["period"])
t0 = np.median(trace["t0"])

# Plot the folded data
# x_fold = (time[mask] - t0 + 0.5*p) % p - 0.5*p
# plt.plot(x_fold, flux[mask] - gp_mod, ".k", label="data", zorder=-1000)
#
# # Overplot the phase binned light curve
# bins = np.linspace(-0.41, 0.41, 50)
# denom, _ = np.histogram(x_fold, bins)
# num, _ = np.histogram(x_fold, bins, weights=flux[mask])
# denom[num == 0] = 1.0
# plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, "o", color="C2",
#          label="binned")
#
# # Plot the folded model
# inds = np.argsort(x_fold)
# inds = inds[np.abs(x_fold)[inds] < 0.3]
# pred = trace["light_curves"][:, inds, 0]
# pred = np.percentile(pred, [16, 50, 84], axis=0)
# plt.plot(x_fold[inds], pred[1], color="C1", label="model")
# art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
#                        zorder=1000)
# art.set_edgecolor("none")
#
# # Annotate the plot with the planet's period
# txt = "period = {0:.5f} +/- {1:.5f} d".format(
#     np.mean(trace["period"]), np.std(trace["period"]))
# plt.annotate(txt, (0, 0), xycoords="axes fraction",
#              xytext=(5, 5), textcoords="offset points",
#              ha="left", va="bottom", fontsize=12)
#
# plt.legend(fontsize=10, loc=4)
# plt.xlim(-0.5*p, 0.5*p)
# plt.xlabel("time since transit [days]")
# plt.ylabel("de-trended flux")
# plt.xlim(-0.15, 0.15)
# plt.savefig("fold.pdf")
# # plt.show()
# plt.clf()
# plt.close()

# And a corner plot of some of the key parameters:

all_pars = ["logw0", "logpower", "logs2", "r_pl", "b", "t0", "logP", "r_star",
            "m_star", "u_star", "mean"]
all_samples = pm.trace_to_dataframe(trace, varnames=all_pars)

print(trace.varnames)

# Convert the radius to Earth radii
# all_samples["r_pl"] = (np.array(all_samples["r_pl"]) * u.R_sun).to(u.R_earth).value
# all_samples["Per"] = np.exp(all_samples["logP"])

all_samples.to_csv("samples.csv", index=False)

for param in all_samples.columns:
    print(param, np.median(all_samples[param]),
          np.abs(np.median(all_samples[param]) - np.percentile(all_samples[param], [50.-68.27/2., 50.+68.27/2.])),
          np.abs(np.median(all_samples[param]) - np.percentile(all_samples[param], [50.-95.45/2., 50.+95.45/2.])))
    plt.hist(all_samples[param], bins=50, color="k")
    _ = [plt.axvline(v) for v in np.percentile(all_samples[param],
                                               [50.-95.45/2., 50.-68.27/2., 50., 50.+68.27/2, 50.+95.45/2.])]
    plt.savefig(param + ".pdf")
    plt.clf()
    plt.close()

corner.corner(all_samples, bins=20)
plt.savefig("corner.pdf")
# plt.show()

# These all seem consistent with the previously published values and an earlier inconsistency between this radius
# measurement and the literature has been resolved by fixing a bug in *exoplanet*.
