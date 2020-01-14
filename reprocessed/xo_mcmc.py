import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
import my_constants as myc
import pymc3 as pm
import theano.tensor as tt

# priors
st_r, st_r_u = 0.455, 0.036
st_m, st_m_u = 0.461, 0.015

t0, t0_u = 3004.8659, 0.01
per, per_u = 26.5837, 0.005
rp, rp_u = 0.0372, 0.005
inc, inc_u = 88.8, 1.
ecc, ecc_u = 0., 0.8
w, w_u = 0., 360.

qld = [0.5079, 0.2239]

# light curve
t, f, e = np.loadtxt("lc-mySFF-cut-transits.dat", unpack=True)

# f -= 1.     # mean -> zero
# f *= 1e3
yerr = e[0]

# plt.plot(t, f, ".")
# plt.show()

# orbit = xo.orbits.KeplerianOrbit(period=per)
# light_curve = xo.StarryLightCurve(u).get_light_curve(orbit=orbit, r=0.1, t=t, texp=0.02).eval()


def lnlike(transit_model, err, lc):
    """ log likelihood function """
    log2pi = np.log(2.0 * np.pi)

    # calculate the lnlike for transit
    sigma = err ** 2.
    chi = np.log(sigma) / 2. + (lc - transit_model) ** 2 / (2. * sigma)

    log_like = - 0.5 * lc.size * log2pi - sum(chi)

    if not np.isfinite(log_like):
        return -np.inf
    else:
        return log_like


with pm.Model() as model:
    # The baseline flux
    # mean = pm.Normal(name="mean", mu=1., sd=0.01)
    mean = 1.

    # Stellar params
    # u_star = xo.distributions.QuadLimbDark("u_star")
    u_star = np.asarray(qld)

    # r_star = pm.Normal(name="r_star", mu=st_r, sd=st_r_u)
    # m_star = pm.Normal(name="m_star", mu=st_m, sd=st_m_u)
    # pm.Potential("r_star_prior", tt.switch(r_star > 0., 0, -np.inf))
    # pm.Potential("m_star_prior", tt.switch(m_star > 0., 0, -np.inf))
    r_star, m_star = st_r, st_m

    # Planet params
    logP = pm.Normal(name="logP", mu=np.log(per), sd=0.002)
    T0 = pm.Normal(name="t0", mu=t0, sd=t0_u)
    r, b = xo.distributions.get_joint_radius_impact(min_radius=0.01, max_radius=0.2,
                                                    testval_r=rp, testval_b=0.9)
    # ecc = pm.Beta("ecc", alpha=0.867, beta=3.03, testval=0.1)
    # omega = xo.distributions.Angle("omega")

    # This shouldn't make a huge difference, but I like to put a uniform prior on the *log* of the radius ratio instead
    # of the value. This can be implemented by adding a custom "potential" (log probability).
    # pm.Potential("r_prior", -pm.math.log(r))
    # pm.Potential("b_prior1", tt.switch(b < 0.98, 0, -np.inf))
    # pm.Potential("b_prior2", tt.switch(b > 0.7, 0, -np.inf))

    period = pm.Deterministic("period", tt.exp(logP))   # period in days (exp ln P)
    # r_pl = pm.Deterministic("r_pl", r)         # planet radius in stellar units

    # Set up a Keplerian orbit for the planets
    orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=T0, b=b)  # ecc=ecc, omega=omega)

    # Compute the model light curve using starry
    light_curves = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit, r=r, t=t, texp=29.4/60./24.,
                                                               oversample=15) + mean
    # light_curve = pm.math.sum(light_curves, axis=-1) + mean
    light_curve = light_curves.flatten()
    pm.Deterministic("light_curves", light_curves)

    # GP model for the light curve
    # kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1. / np.sqrt(2.))
    # gp = xo.gp.GP(kernel, t, tt.exp(logs2), J=2)
    # pm.Potential("transit_obs", gp.log_likelihood(f - light_curve))
    # pm.Deterministic("gp_pred", gp.predict())

    pm.Normal(name="obs", mu=light_curve, sd=yerr, observed=f)

    for key in model.__dict__.keys():
        print key, model.__dict__[key]

    start = model.test_point
    map_soln = pm.find_MAP(start=start, vars=[model.t0])
    map_soln = pm.find_MAP(start=map_soln, vars=[model.logP])
    map_soln = pm.find_MAP(start=map_soln, vars=[model.rb])
    map_soln = pm.find_MAP(start=map_soln)

    # map_soln = pm.find_MAP(start=start, vars=[logs2, logS0, logw0])
    # map_soln = pm.find_MAP(start=start, vars=[model.rb])
    # map_soln = pm.find_MAP(start=map_soln)

lc = map_soln["light_curves"].flatten()

# for key in map_soln.keys():
#     print key, map_soln[key]

# plt.plot(t, f, ".k", ms=4)
# plt.plot(t, lc, lw=1)
# plt.xlim(t.min(), t.max())
# plt.ylabel("relative flux")
# plt.xlabel("time [days]")
# plt.show()

sampler = xo.PyMC3Sampler(start=200, window=100, finish=200)
with model:
    burnin = sampler.tune(tune=3000, start=map_soln, step_kwargs=dict(target_accept=0.9))     # step_kwargs=dict(target_accept=0.9)
with model:
    trace = sampler.sample(draws=2000)

print trace.varnames
print
print pm.summary(trace, varnames=["period", "t0", "r", "b"])

import corner
samples = pm.trace_to_dataframe(trace, varnames=["period", "t0", "r", "b"])
print samples
samples["r_pl"] = np.array(samples["r__0"]) * st_r * myc.RS / myc.RE
corner.corner(samples)
plt.show()


p = np.median(trace["period"])
t0 = np.median(trace["t0"])

# Plot the folded data
x_fold = (t - t0 + 0.5*p) % p - 0.5*p
plt.plot(x_fold, f, ".k", label="data", zorder=-1000)

# Plot the folded model
inds = np.argsort(x_fold)
pred = trace["light_curves"][:, inds, 0]
pred = np.percentile(pred, [16, 50, 84], axis=0)
plt.plot(x_fold[inds], pred[1], color="C1", label="model")
art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5, zorder=1000)
art.set_edgecolor("none")

# Annotate the plot with the planet's period
txt = "period = {0:.5f} +/- {1:.5f} d".format(np.mean(trace["period"]), np.std(trace["period"]))
plt.annotate(txt, (0, 0), xycoords="axes fraction", xytext=(5, 5), textcoords="offset points",
             ha="left", va="bottom", fontsize=12)

plt.legend(fontsize=10, loc=4)
plt.xlim(-0.5*p, 0.5*p)
plt.xlabel("time since transit [days]")
plt.ylabel("de-trended flux")
plt.xlim(-0.15, 0.15)
plt.show()
