"""
LC MCMC fitting for 4 planets

- subtract models of other planets in plots
- try K2SC-masked, K2SFF, Everest, etc - mask outliers
- make individual 4x4 corner plots

- fit/update limb darkening?
- fit_ecc needs w as a free parameter
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import emcee
import batman
import sys
from exoplanet import phase_fold
# import astropy.io.ascii as ascii
# import scipy.interpolate as interpolate
# import pandas
import corner
import dill
# from multiprocessing import Pool
import my_constants as myc
import os

st_r, st_m = 0.46, 0.50

pl_use = [True, True, True, True]     # only run for these planets - limited to d,c,b,01 for now

# set priors and widths
prior_t0 = np.array([2457821.3168, 2457823.7656, 2457826.1739, 2457837.8659]) - 2454833.
width_t0 = [0.01, 0.01, 0.01, 0.01]
prior_per = [3.0712, 4.8682, 11.0234, 26.5837]
width_per = [0.001, 0.001, 0.001, 0.01]
prior_rp = [0.0255, 0.0288, 0.0393, 0.0372]
width_rp = [0.005, 0.005, 0.005, 0.001]
prior_i = [86., 87., 88., 88.5]
width_i = [4., 3., 2., 1.5]

pl_chars = ["b", "c", "d", "01"]


def single_model(pars, t):
    """ return the transit curve at times t """

    rp, inc, t0, per = pars

    params = batman.TransitParams()
    params.limb_dark = "quadratic"  # limb darkening model
    params.u = [0.5079, 0.2239]     # limb darkening coefficients
    params.ecc = 0.
    params.w = 0.

    params.rp = rp
    params.inc = inc
    params.t0 = t0
    params.per = per
    params.a = keplerslaw(per)  # set from keplers law and stellar relations

    batman_model = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.)
    light_curve = batman_model.light_curve(params)

    return light_curve


def calc_i(_a, _b):
    return np.degrees(np.arccos(_b / _a))


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


def keplerslaw(kper):
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
        self.ecc = 0.                   # eccentricity
        self.w = 90.                    # longitude of periastron (in degrees)

        # free parameters
        if pl_use[2]:
            self.rp_d = 0.          # planet radius (in units of stellar radii)
            self.a_d = 0.           # semi-major axis (in units of stellar radii)
            self.inc_d = 0.         # orbital inclination (in degrees)
            self.t0_d = 0.          # time of inferior conjunction
            self.period_d = 0.      # orbital period
        if pl_use[1]:
            self.rp_c = 0.          # planet radius (in units of stellar radii)
            self.a_c = 0.           # semi-major axis (in units of stellar radii)
            self.inc_c = 0.         # orbital inclination (in degrees)
            self.t0_c = 0.          # time of inferior conjunction
            self.period_c = 0.      # orbital period
        if pl_use[0]:
            self.rp_b = 0.          # planet radius (in units of stellar radii)
            self.a_b = 0.           # semi-major axis (in units of stellar radii)
            self.inc_b = 0.         # orbital inclination (in degrees)
            self.t0_b = 0.          # time of inferior conjunction
            self.period_b = 0.      # orbital period
        if pl_use[3]:
            self.rp_01 = 0.          # planet radius (in units of stellar radii)
            self.a_01 = 0.           # semi-major axis (in units of stellar radii)
            self.inc_01 = 0.         # orbital inclination (in degrees)
            self.t0_01 = 0.          # time of inferior conjunction
            self.period_01 = 0.      # orbital period

        self.N_free_parameters = sum(pl_use) * 4     # number of free parameters in model

        # start the batman model
        self.params = batman.TransitParams()
        self.params.limb_dark = self.limb_dark  # limb darkening model
        self.params.u = self.u                  # limb darkening coefficients

        # names of the parameters for the corner plot
        self.labels = []
        if pl_use[2]:
            self.labels += [r'$R_p/R_* d$', r'$i d$', r'$T_0 d$', r'$P d$']
        if pl_use[1]:
            self.labels += [r'$R_p/R_* c$', r'$i c$', r'$T_0 c$', r'$P c$']
        if pl_use[0]:
            self.labels += [r'$R_p/R_* b$', r'$i b$', r'$T_0 b$', r'$P b$']
        if pl_use[3]:
            self.labels += [r'$R_p/R_* 01$', r'$i 01$', r'$T_0 01$', r'$P 01$']

    def set_transit_parameters(self, pars):
        """ set transit parameters from pars """
        if pl_use[2]:
            self.rp_d, self.inc_d, self.t0_d, self.period_d = pars[:4]
        if pl_use[1]:
            self.rp_c, self.inc_c, self.t0_c, self.period_c = pars[4:8]
        if pl_use[0]:
            self.rp_b, self.inc_b, self.t0_b, self.period_b = pars[8:12]
        if pl_use[3]:
            self.rp_01, self.inc_01, self.t0_01, self.period_01 = pars[12:]

        if pl_use[2]:
            self.a_d = keplerslaw(np.asarray(self.period_d))  # set from keplers law and stellar relations
        if pl_use[1]:
            self.a_c = keplerslaw(np.asarray(self.period_c))
        if pl_use[0]:
            self.a_b = keplerslaw(np.asarray(self.period_b))
        if pl_use[3]:
            self.a_01 = keplerslaw(np.asarray(self.period_01))

    def get_transit_curve(self, t):
        """ return the transit curve at times t """
        light_curve = np.ones(t.size, dtype=float)

        # set self.params for batman model
        self.params.ecc = self.ecc
        self.params.w = self.w

        if pl_use[2]:
            self.params.rp = self.rp_d
            self.params.inc = self.inc_d
            self.params.t0 = self.t0_d
            self.params.per = self.period_d
            self.params.a = keplerslaw(self.period_d)  # set from keplers law and stellar relations
            self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=29.4/60./24.)
            light_curve += self.batman_model.light_curve(self.params) - 1.
        if pl_use[1]:
            self.params.rp = self.rp_c
            self.params.inc = self.inc_c
            self.params.t0 = self.t0_c
            self.params.per = self.period_c
            self.params.a = keplerslaw(self.period_c)
            self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=29.4/60./24.)
            light_curve += self.batman_model.light_curve(self.params) - 1.
        if pl_use[0]:
            self.params.rp = self.rp_b
            self.params.inc = self.inc_b
            self.params.t0 = self.t0_b
            self.params.per = self.period_b
            self.params.a = keplerslaw(self.period_b)
            self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=29.4/60./24.)
            light_curve += self.batman_model.light_curve(self.params) - 1.
        if pl_use[3]:
            self.params.rp = self.rp_01
            self.params.inc = self.inc_01
            self.params.t0 = self.t0_01
            self.params.per = self.period_01
            self.params.a = keplerslaw(self.period_01)
            self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=29.4/60./24.)
            light_curve += self.batman_model.light_curve(self.params) - 1.

        return light_curve

    def set_priors(self):
        """ define each parameter prior distribution """
        if pl_use[2]:
            self.prior_rp_d = stats.norm(prior_rp[2], width_rp[2])    # planet/star radius ratio
            self.prior_inc_d = stats.uniform(prior_i[2], width_i[2])  # orbital inclination (degrees)
            self.prior_t0_d = stats.norm(prior_t0[2], width_t0[2])    # time of inferior conjunction
            self.prior_period_d = stats.norm(prior_per[2], width_per[2])  # orbital period (days)
        if pl_use[1]:
            self.prior_rp_c = stats.norm(prior_rp[1], width_rp[1])    # planet/star radius ratio
            self.prior_inc_c = stats.uniform(prior_i[1], width_i[1])  # orbital inclination (degrees)
            self.prior_t0_c = stats.norm(prior_t0[1], width_t0[1])    # time of inferior conjunction
            self.prior_period_c = stats.norm(prior_per[1], width_per[1])  # orbital period (days)
        if pl_use[0]:
            self.prior_rp_b = stats.norm(prior_rp[0], width_rp[0])    # planet/star radius ratio
            self.prior_inc_b = stats.uniform(prior_i[0], width_i[0])  # orbital inclination (degrees)
            self.prior_t0_b = stats.norm(prior_t0[0], width_t0[0])    # time of inferior conjunction
            self.prior_period_b = stats.norm(prior_per[0], width_per[0])  # orbital period (days)
        if pl_use[3]:
            self.prior_rp_01 = stats.norm(prior_rp[3], width_rp[3])    # planet/star radius ratio
            self.prior_inc_01 = stats.uniform(prior_i[3], width_i[3])  # orbital inclination (degrees)
            self.prior_t0_01 = stats.norm(prior_t0[3], width_t0[3])    # time of inferior conjunction
            self.prior_period_01 = stats.norm(prior_per[3], width_per[3])  # orbital period (days)

    def get_from_prior(self, nwalkers):
        """ return a list with random values from each parameter's prior """
        self.set_priors()   # use distributions from set_priors

        pfp = []
        if pl_use[2]:
            pfp += [self.prior_rp_d.rvs(nwalkers), self.prior_inc_d.rvs(nwalkers), self.prior_t0_d.rvs(nwalkers),
                    self.prior_period_d.rvs(nwalkers)]
        if pl_use[1]:
            pfp += [self.prior_rp_c.rvs(nwalkers), self.prior_inc_c.rvs(nwalkers), self.prior_t0_c.rvs(nwalkers),
                    self.prior_period_c.rvs(nwalkers)]
        if pl_use[0]:
            pfp += [self.prior_rp_b.rvs(nwalkers), self.prior_inc_b.rvs(nwalkers), self.prior_t0_b.rvs(nwalkers),
                    self.prior_period_b.rvs(nwalkers)]
        if pl_use[3]:
            pfp += [self.prior_rp_01.rvs(nwalkers), self.prior_inc_01.rvs(nwalkers), self.prior_t0_01.rvs(nwalkers),
                    self.prior_period_01.rvs(nwalkers)]

        pars_from_prior = np.asarray(pfp).T

        return pars_from_prior


class Data(object):
    # TODO: Masking

    """ GLOBAL class to hold the light curve """
    def __init__(self, lc_file, skip_lc_rows=0):
        self.lc_file = lc_file

        # read light curve
        self.LCtime, self.LC, self.LCerror = np.loadtxt(lc_file, unpack=True, skiprows=skip_lc_rows, delimiter=",")

        # if epic == '248690431':
        #     mask = []
        #     self.LCtime = np.delete(self.LCtime, mask)
        #     self.LC = np.delete(self.LC, mask)
        #     self.LCerror = np.delete(self.LCerror, mask)

        self.N_lc = self.LCtime.size

        # plt.plot(self.LCtime, self.LC, ".")
        # plt.show()


def lnlike(pars, planet):
    """ log likelihood function """
    log2pi = np.log(2.0*np.pi)

    # set the transit params
    planet.set_transit_parameters(pars)

    # calculate the lnlike for transit
    transit_model = planet.get_transit_curve(data.LCtime)
    sigma = data.LCerror**2
    chi = np.log(sigma)/2. + (data.LC - transit_model)**2 / (2.*sigma)
    
    log_like_transit = - 0.5*data.N_lc*log2pi - np.sum(chi)
    
    log_like = log_like_transit

    if not np.isfinite(log_like):  # or any(pars) < 0. or any(pars[::4]) > 90.:
        return -np.inf
    else:
        return log_like


def lnprior(pars, planet):
    """ calculate the log prior for a set of parameters """
    ln_prior = 0.
    # transit parameters
    if pl_use[2]:
        prior_rp_d = planet.prior_rp_d.logpdf(pars[0])
        prior_inc_d = planet.prior_inc_d.logpdf(pars[1])
        prior_t0_d = planet.prior_t0_d.logpdf(pars[2])
        prior_period_d = planet.prior_period_d.logpdf(pars[3])
        ln_prior_d = prior_rp_d + prior_inc_d + prior_t0_d + prior_period_d
        ln_prior += ln_prior_d
    if pl_use[1]:
        prior_rp_c = planet.prior_rp_c.logpdf(pars[4])
        prior_inc_c = planet.prior_inc_c.logpdf(pars[5])
        prior_t0_c = planet.prior_t0_c.logpdf(pars[6])
        prior_period_c = planet.prior_period_c.logpdf(pars[7])
        ln_prior_c = prior_rp_c + prior_inc_c + prior_t0_c + prior_period_c
        ln_prior += ln_prior_c
    if pl_use[0]:
        prior_rp_b = planet.prior_rp_b.logpdf(pars[8])
        prior_inc_b = planet.prior_inc_b.logpdf(pars[9])
        prior_t0_b = planet.prior_t0_b.logpdf(pars[10])
        prior_period_b = planet.prior_period_b.logpdf(pars[11])
        ln_prior_b = prior_rp_b + prior_inc_b + prior_t0_b + prior_period_b
        ln_prior += ln_prior_b
    if pl_use[3]:
        prior_rp_01 = planet.prior_rp_01.logpdf(pars[12])
        prior_inc_01 = planet.prior_inc_01.logpdf(pars[13])
        prior_t0_01 = planet.prior_t0_01.logpdf(pars[14])
        prior_period_01 = planet.prior_period_01.logpdf(pars[15])
        ln_prior_01 = prior_rp_01 + prior_inc_01 + prior_t0_01 + prior_period_01
        ln_prior += ln_prior_01

    return ln_prior


def lnprob(pars, planet):
    """ posterior distribution """
    log_prior = lnprior(pars, planet)
    log_like = lnlike(pars, planet)
    log_posterior = log_prior + log_like

    return log_posterior


# for i in range(4):
#     print(calc_i(keplerslaw(prior_per[i]), 1.+prior_rp[i]))
# sys.exit()

# initialize the Planet and Data classes
planet = Planet()

method = ["K2SC", "K2SFF"][0]
data = Data("LC_{}.dat".format(method))

# parameters for emcee
# ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, planet.N_free_parameters*3, 100, 40      # testing
# ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 60, 300, 100      # basic
ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 250, 2000, 1000    # fitting

out_dir = "save_{}_{}_{}_{}_{}/".format(method, ndim, nwalkers, nsteps, burnin)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# get random starting positions from the priors
pos = planet.get_from_prior(nwalkers)
# print "Priors between", [(min(pri), max(pri)) for pri in np.asarray(pos).T]

# sample the posterior
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(planet,))
out = sampler.run_mcmc(pos, nsteps, progress=True)

# pool = Pool()   # use 4 cores for ~2 times speed
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(planet,), pool=pool)
# out = sampler.run_mcmc(pos, nsteps, progress=True)

# remove some of the initial steps (burn-in)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# make a corner plot of the MCMC samples
corner.corner(samples, labels=planet.labels)
plt.savefig(out_dir+"samples_all.pdf", format="pdf")
# plt.show()
plt.close()

# get the medians of the posterior distributions
median_pars = np.median(samples, axis=0)

print(['%9s' % s.replace('$', '').replace('_', '').replace('\\rm', '').replace('\\', '').replace(' d', '')
       for s in planet.labels[:4]])
print(np.asarray(['%9.4f' % s for s in median_pars]).reshape((4, planet.N_free_parameters/4)))
pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))

save_str = ""
for pc in pcs:
    # vals = list(pc)
    # save_str += (",".join(vals) + "\n")
    save_str += (str(pc) + "\n")
with open(out_dir+"fit_vals.dat", "w") as save_file:
    save_file.write(save_str)

planet.set_transit_parameters(median_pars)
full_model = planet.get_transit_curve(data.LCtime)

for i in range(4):
    if pl_use[[2, 1, 0, 3][i]]:
        if i == 0:
            pack = ["d", planet.period_d, planet.t0_d, planet.rp_d, planet.inc_d, median_pars[:4], [None, 4]]
        if i == 1:
            pack = ["c", planet.period_c, planet.t0_c, planet.rp_c, planet.inc_c, median_pars[4:8], [4, 8]]
        if i == 2:
            pack = ["b", planet.period_b, planet.t0_b, planet.rp_b, planet.inc_b, median_pars[8:12], [8, 12]]
        if i == 3:
            pack = ["01", planet.period_01, planet.t0_01, planet.rp_01, planet.inc_01, median_pars[12:], [12, None]]
        pl, period, t0, rp, inc, pars, inds = pack

        a = keplerslaw(period)
        b = calc_b(a, inc)
        rp = rp * st_r * myc.RS / myc.RE
        print("\nPlanet %s:\nb = %.2f\nRp = %.2f RE\na = %.2f sr" % (pl, b, rp, a))
        print(pars)

        t = np.linspace(data.LCtime.min(), data.LCtime.max(), data.LCtime.size*100)     # supersample
        lc = single_model(pars, t)

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3, 1]})

        # plot full LC and single model
        ax1.plot(data.LCtime, data.LC, 'o', ms=3, alpha=0.7, zorder=1)
        ax1.plot(t, lc, lw=2, alpha=0.8, zorder=2)
        ax1.set_xlim([min(data.LCtime), max(data.LCtime)])

        # plot phase-folded LC and ss-single model
        phase, lcfold = phase_fold(data.LCtime, data.LC, period, t0)
        phase = np.asarray(phase)
        phase = (phase - 0.5) * period * 24.    # hours from transit centre
        ax2.plot(phase, lcfold, 'o', ms=3, alpha=0.7, zorder=1)
        mphase, mfold = phase_fold(t, lc, period, t0)
        mphase = np.asarray(mphase)
        mphase = (mphase - 0.5) * period * 24.
        ax2.plot(mphase, mfold, lw=2, alpha=0.8, zorder=2)
        ax2.set_xlim(-3, 3)
        ax2.tick_params(axis='x', labelbottom=False)

        # plot phase-folded residuals
        _, mfold_lct = phase_fold(data.LCtime, single_model(pars, data.LCtime), period, t0)
        resid = np.asarray(lcfold) - np.asarray(mfold_lct)
        ax3.plot(phase, resid, 'o', alpha=0.7, ms=3, zorder=1)
        ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=2)
        ax3.set_xlim(-3, 3)
        ax3.set_ylim([-np.std(resid)*5., np.std(resid)*5.])

        ax2.get_shared_x_axes().join(ax2, ax3)
        plt.tight_layout()
        plt.savefig(out_dir+"fit_{}.pdf".format(pl), format="pdf")
        # plt.show()
        plt.close()

        corner.corner(samples[:,inds[0]:inds[1]], labels=planet.labels[inds[0]:inds[1]])
        plt.savefig(out_dir+"samples_{}.pdf".format(pl), format="pdf")
        plt.close()

# mcmc_cols = ["mcmc_rp", "mcmc_inc", "mcmc_t0", "mcmc_period"]
# # if fit_ecc:
# #     mcmc_cols += ["mcmc_ecc"]
# for i in range(planet.N_free_parameters):
#     col = mcmc_cols[i]
#     pf.iloc[rowid][col] = "%.4f|%.4f|%.4f" % pcs[i]
# pf.iloc[rowid]["mcmc_rj"] = round(rp, 2)
# pf.iloc[rowid]["mcmc_rs"] = round(rs, 2)
# pf.iloc[rowid]["mcmc_b"] = round(b, 2)
# pf.to_csv("../mons_fitting.csv", index=False)

with open(out_dir+"mcmc.pkl", "wb") as pklf:
    dill.dump([data, planet, samples], pklf)
