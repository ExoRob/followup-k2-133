import platform
on_kelvin = 'kelvin' in platform.node()
if on_kelvin:
    print "> Running on Kelvin ..."
    import matplotlib
    matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import emcee
import batman
import sys
from my_exoplanet import phase_fold
import my_constants as myc
import astropy.io.ascii as ascii
import scipy.interpolate as interpolate
# import pandas
import corner
import dill
# from pathos.multiprocessing import Pool
from tqdm import tqdm
from time import time

Rsun = myc.RS
Rjup = myc.RJ

st_r, st_m = 0.455, 0.461

# pf = pandas.read_csv("fitting.csv", delimiter=",", dtype=str)
# rowid = pf.planet.tolist().index("e")
# row = pf.iloc[rowid]

fit_ecc = False


def calc_i(_a, _b):
    return np.degrees(np.arccos(_b / _a))


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


t01,t02 = 3004.8659, 0.005
p1,p2 = 26.5837, 0.005
rp1,rp2 = 0.0372, 0.015
i1,i2 = 89., .3
# e1, e2 = 0.0001, 0.8
# w1, w2 = 0.0001, 359.9999


def keplerslaw(kper, st_r, st_m):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


class Planet(object):
    def __init__(self):
        # fixed parameters
        self.limb_dark = "quadratic"    # limb darkening model
        self.u = [0.5079, 0.2239]       # limb darkening coefficients
        self.ecc = 0.       # eccentricity
        self.w = 0.         # longitude of periastron (in degrees)

        # stellar parameters
        self.stellar_radius = st_r
        self.stellar_mass = st_m

        # free parameters
        self.rp = 0.          # planet radius (in units of stellar radii)
        self.a = 0.           # semi-major axis (in units of stellar radii)
        self.inc = 0.         # orbital inclination (in degrees)
        self.t0 = 0.          # time of inferior conjunction
        self.period = 0.      # orbital period
        self.N_free_parameters = 4 + fit_ecc * 2     # number of free parameters in model

        # start the batman model
        self.params = batman.TransitParams()
        self.params.limb_dark = self.limb_dark  # limb darkening model
        self.params.u = self.u                  # limb darkening coefficients

        # names of the parameters for the corner plot
        self.labels = [r'$R_p/R_*$', r'$i$', r'$T_0$', r'$P$']
        if fit_ecc:
            self.labels += [r'$e$', r'$w$']

    def set_transit_parameters(self, pars):
        """ set transit parameters from pars """
        self.rp, self.inc, self.t0, self.period = pars[:4]
        if fit_ecc:
            self.ecc, self.w = pars[-2:]
        self.a = keplerslaw(self.period, self.stellar_radius, self.stellar_mass)
    
    def get_transit_curve(self, t):
        """ return the transit curve at times t """
        # set self.params for batman model
        self.params.rp = self.rp
        self.params.inc = self.inc
        self.params.t0 = self.t0
        self.params.per = self.period
        self.params.ecc = self.ecc
        self.params.w = self.w
        self.params.a = keplerslaw(self.period, self.stellar_radius, self.stellar_mass)

        self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=0.5/24.)
        light_curve = self.batman_model.light_curve(self.params)
        
        return light_curve

    def set_priors(self):
        """ define each parameter prior distribution """
        # rp1,rp2 = [float(x) for x in row.prior_rp.split("|")]
        # i1,i2 = [float(x) for x in row.prior_inc.split("|")]
        # t01,t02 = [float(x) for x in row.prior_t0.split("|")]
        # p1,p2 = [float(x) for x in row.prior_period.split("|")]
        # e1, e2 = [float(x) for x in row.prior_ecc.split("|")]
        # w1, w2 = [float(x) for x in row.prior_w.split("|")]
        self.prior_rp = stats.norm(rp1, rp2)    # planet radius (in units of stellar radii)
        self.prior_inc = stats.uniform(i1, i2)  # orbital inclination (in degrees)
        self.prior_t0 = stats.norm(t01, t02)    # time of inferior conjunction (in BJD)
        self.prior_period = stats.norm(p1, p2)  # orbital period (in days)
        if fit_ecc:
            self.prior_ecc = stats.uniform(e1, e2)  # eccentricity
            self.prior_w = stats.uniform(w1, w2)

    def get_from_prior(self, nwalkers):
        """ return a list with random values from each parameter's prior """
        self.set_priors()   # use distributions from set_priors

        pfp = [self.prior_rp.rvs(nwalkers), self.prior_inc.rvs(nwalkers), self.prior_t0.rvs(nwalkers),
               self.prior_period.rvs(nwalkers)]
        if fit_ecc:
            pfp += [self.prior_ecc.rvs(nwalkers), self.prior_w.rvs(nwalkers)]
        pars_from_prior = np.asarray(pfp).T

        # with open("pkl.pkl", "wb") as pkf:
        #     dill.dump(pars_from_prior, pkf)
        # sys.exit()

        return pars_from_prior


class Data(object):
    """ GLOBAL class to hold the light curve """
    def __init__(self, lc_file, skip_lc_rows=0):
        self.lc_file = lc_file

        # read light curve
        self.LCtime, self.LC, self.LCerror = np.loadtxt(lc_file, unpack=True, skiprows=skip_lc_rows)

        self.N_lc = self.LCtime.size


def lnlike(pars, planet):
    """ log likelihood function """
    log2pi = np.log(2.*np.pi)

    # set the transit params
    planet.set_transit_parameters(pars)

    # calculate the lnlike for transit
    transit_model = planet.get_transit_curve(data.LCtime)
    sigma = data.LCerror**2
    chi = np.log(sigma)/2. + (data.LC - transit_model)**2 / (2.*sigma)
    
    log_like_transit = - 0.5*data.N_lc*log2pi - np.sum(chi)
    
    # if you want, try to calculate lnlike using one of the scipy distributions
    # log_like_transit2 = stats.norm(loc=transit_model, scale=sigma).logpdf(data.LC).sum()

    # the total log likelihood (neglect jitter)
    log_like = log_like_transit

    if (not np.isfinite(log_like)) or (any(pars < 0.0)):
        return -np.inf
    elif fit_ecc and (pars[-2] > e2 or pars[-1] > w2):
        return -np.inf
    else:
        return log_like


def lnprior(pars, planet):
    """ calculate the log prior for a set of parameters """
    # transit parameters
    prior_rp = planet.prior_rp.logpdf(pars[0])
    prior_inc = planet.prior_inc.logpdf(pars[1])
    prior_t0 = planet.prior_t0.logpdf(pars[2])
    prior_period = planet.prior_period.logpdf(pars[3])

    ln_prior = prior_rp + prior_inc + prior_t0 + prior_period
    if fit_ecc:
        prior_ecc = planet.prior_ecc.logpdf(pars[-2])
        prior_w = planet.prior_w.logpdf(pars[-1])
        ln_prior += prior_ecc
        ln_prior += prior_w

    return ln_prior


# ofile = open("probs.dat", "wa")

def lnprob(pars, planet):
    """ posterior distribution """
    log_prior = lnprior(pars, planet)
    log_like = lnlike(pars, planet)
    log_posterior = log_prior + log_like

    # all_lnprobs.append([list(pars), log_prior, log_like])
    # ofile.write(str([list(pars), log_prior, log_like]) + "\n")

    return log_posterior


# initialize the Planet and Data classes
planet = Planet()
data = Data(lc_file="final-lc-mySFF-cut-transits.dat")

# parameters for emcee
# ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 50, 150, 60       # testing
# ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 100, 1000, 500     # basic
ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 300, 15000, 10000   # fitting
# ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 800, 100000, 85000     # long

# get random starting positions from the priors
pos = planet.get_from_prior(nwalkers)
print pos.shape, ndim, nwalkers, nsteps, burnin
# print "Priors between", [(min(pri), max(pri)) for pri in np.asarray(pos).T]

# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(planet,))
# t1 = time()
# for pos,lnp,rstate in tqdm(sampler.sample(pos, iterations=nsteps)):
#     t2 = time()
#
#     if (t2 - t1) > 10.:     # if it hangs
#         with open("ln_probs.pkl", "wb") as pf:
#             dill.dump(all_lnprobs, pf)
#         sys.exit()
#     else:
#         t1 = t2

# This now uses pathos !!
# pool = Pool(processes=1)   # use 4 cores for ~2 times speed
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(planet,), pool=None)
out = sampler.run_mcmc(pos, nsteps, progress=True)

# remove burn-in
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# make a corner plot
corner.corner(samples, labels=planet.labels, bins=50)
plt.savefig("samples-short.pdf", format="pdf")
# plt.show()
plt.close("all")

# get the medians of the posterior distributions
median_pars = np.median(samples, axis=0)

print ['%9s' % s.replace('$', '').replace('_', '').replace('\\rm', '').replace('\\', '') for s in planet.labels]
print ['%9.4f' % s for s in median_pars]
pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))
print pcs

planet.set_transit_parameters(median_pars)

rs, ms = planet.stellar_radius, planet.stellar_mass
b = calc_b(keplerslaw(planet.period, planet.stellar_radius, planet.stellar_mass), planet.inc)
a = keplerslaw(planet.period, planet.stellar_radius, planet.stellar_mass)
rp = planet.rp * rs * Rsun / Rjup

print "\nb = %.2f\nRp = %.2f RJ\nRs = %.2f RS" % (b, rp, rs)
print "a = %.2f sr" % keplerslaw(planet.period, planet.stellar_radius, planet.stellar_mass)

t = np.linspace(data.LCtime.min(), data.LCtime.max(), data.LCtime.size*100)
lc = planet.get_transit_curve(t)

fig, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(14, 6), gridspec_kw={'height_ratios': [4, 2]})
# ax1.plot(data.LCtime, data.LC, 'o', ms=3, alpha=0.7, zorder=1)
# ax1.plot(t, lc, lw=2, alpha=0.8, zorder=2)
# ax1.set_xlim([min(data.LCtime), max(data.LCtime)])

phase, lcfold = phase_fold(data.LCtime, data.LC, planet.period, planet.t0)
# ax2.plot(phase, lcfold, 'o', ms=3, alpha=0.7, zorder=1)
ax2.errorbar(phase, lcfold, data.LCerror, lw=0., marker=".", ms=10, elinewidth=1.5, zorder=1, color="k")

mphase, mfold = phase_fold(t, lc, planet.period, planet.t0)
ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2)
ax2.set_xlim([0.495, 0.505])
ax2.tick_params(axis='x', labelbottom=False)

_, mfold_lct = phase_fold(data.LCtime, planet.get_transit_curve(data.LCtime), planet.period, planet.t0)
resid = np.asarray(lcfold) - np.asarray(mfold_lct)
ax3.plot(phase, resid, 'o', alpha=0.7, ms=3, zorder=1)
ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=2)
_ = [ax3.axhspan(-data.LCerror[0]*i, data.LCerror[0]*i, color="grey", alpha=0.25) for i in np.arange(1, 4)]
ax3.set_xlim([0.495, 0.505])
ax3.set_ylim([-data.LCerror[0]*3, data.LCerror[0]*3])

ax2.get_shared_x_axes().join(ax2, ax3)
plt.tight_layout()
plt.savefig("fit-short.pdf", format="pdf")
# plt.show()
plt.close("all")

# mcmc_cols = ["mcmc_rp", "mcmc_inc", "mcmc_t0", "mcmc_period"]
# if fit_ecc:
#     mcmc_cols += ["mcmc_ecc", "mcmc_w"]
# for i in range(planet.N_free_parameters):
#     col = mcmc_cols[i]
#     pf.iloc[rowid][col] = "%.4f|%.4f|%.4f" % pcs[i]
# pf.iloc[rowid]["mcmc_rj"] = round(rp, 2)
# pf.iloc[rowid]["mcmc_rs"] = round(rs, 2)
# pf.iloc[rowid]["mcmc_b"] = round(b, 2)
# pf.to_csv("fitting.csv", index=False)

with open("mcmc-short.pkl", "wb") as pklf:
    dill.dump([data, planet, samples], pklf)
