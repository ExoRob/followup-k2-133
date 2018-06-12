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

        self.rp_01 = 0.          # planet radius (in units of stellar radii)
        self.a_01 = 0.           # semi-major axis (in units of stellar radii)
        self.inc_01 = 0.         # orbital inclination (in degrees)
        self.t0_01 = 0.          # time of inferior conjunction
        self.period_01 = 0.      # orbital period

        self.N_free_parameters = 4     # number of free parameters in model

        # start the batman model
        self.params = batman.TransitParams()
        self.params.limb_dark = self.limb_dark  # limb darkening model
        self.params.u = self.u                  # limb darkening coefficients

        # names of the parameters for the corner plot
        self.labels = [r'$R_p/R_*$', r'$i$', r'$T_0$', r'$P$']

    def set_transit_parameters(self, pars):
        """ set transit parameters from pars """
        self.rp_01, self.inc_01, self.t0_01, self.period_01 = pars
        self.a_01 = keplerslaw(np.asarray(self.period_01))

    def get_transit_curve(self, t):
        """ return the transit curve at times t """
        # set self.params for batman model
        self.params.ecc = self.ecc
        self.params.w = self.w

        self.params.rp = self.rp_01
        self.params.inc = self.inc_01
        self.params.t0 = self.t0_01
        self.params.per = self.period_01
        self.params.a = keplerslaw(self.period_01)
        self.batman_model = batman.TransitModel(self.params, t, supersample_factor=15, exp_time=29.4/60./24.)
        light_curve = self.batman_model.light_curve(self.params)

        return light_curve

    def set_priors(self):
        """ define each parameter prior distribution """
        self.prior_rp_01 = stats.norm(prior_rp[3], width_rp[3])    # planet/star radius ratio
        self.prior_inc_01 = stats.uniform(prior_i[3], width_i[3])  # orbital inclination (degrees)
        self.prior_t0_01 = stats.norm(prior_t0[3], width_t0[3])    # time of inferior conjunction
        self.prior_period_01 = stats.norm(prior_per[3], width_per[3])  # orbital period (days)

    def get_from_prior(self, nwalkers):
        """ return a list with random values from each parameter's prior """
        self.set_priors()   # use distributions from set_priors

        pfp = [self.prior_rp_01.rvs(nwalkers), self.prior_inc_01.rvs(nwalkers), self.prior_t0_01.rvs(nwalkers),
               self.prior_period_01.rvs(nwalkers)]

        pars_from_prior = np.asarray(pfp).T

        return pars_from_prior


class Data(object):
    """ GLOBAL class to hold the light curve """
    def __init__(self, lc_file, skip_lc_rows=0, cut=None):
        self.lc_file = lc_file

        # read light curve
        self.LCtime, self.LC = np.loadtxt(lc_file, unpack=True, skiprows=skip_lc_rows, delimiter=",")
        self.LCerror = np.ones(self.LCtime.size, float) * np.std(self.LC)

        if cut:
            self.LCtime = np.delete(self.LCtime, cut)
            self.LC = np.delete(self.LC, cut)
            self.LCerror = np.ones(self.LCtime.size, float) * np.std(self.LC)

        self.N_lc = self.LCtime.size


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
    # transit parameters
    prior_rp_01 = planet.prior_rp_01.logpdf(pars[0])
    prior_inc_01 = planet.prior_inc_01.logpdf(pars[1])
    prior_t0_01 = planet.prior_t0_01.logpdf(pars[2])
    prior_period_01 = planet.prior_period_01.logpdf(pars[3])
    ln_prior_01 = prior_rp_01 + prior_inc_01 + prior_t0_01 + prior_period_01
    ln_prior = ln_prior_01

    return ln_prior


def lnprob(pars, planet):
    """ posterior distribution """
    log_prior = lnprior(pars, planet)
    log_like = lnlike(pars, planet)
    log_posterior = log_prior + log_like

    return log_posterior


in_transit_mask = np.loadtxt("in_transit_mask.dat", unpack=True, dtype=str) == "True"
to_cut = np.argwhere(in_transit_mask).T[0]

st_r, st_m = 0.456, 0.497

# set priors and widths
prior_t0 = np.array([2457821.3168, 2457823.7656, 2457826.1739, 2457837.8659]) - 2454833.
width_t0 = [0.01, 0.01, 0.01, 0.01]
prior_per = [3.0712, 4.8682, 11.0234, 26.5837]
width_per = [0.001, 0.001, 0.001, 0.01]
prior_rp = [0.0255, 0.0288, 0.0393, 0.0372]
width_rp = [0.005, 0.005, 0.005, 0.001]
prior_i = [86., 87., 88., 88.5]
width_i = [4., 3., 2., 1.5]
out_dir = "jackknife/"

t_full, f_full = np.loadtxt("01_lc.dat", unpack=True, delimiter=",")

count = 0
for ind in to_cut:
    # initialize the Planet and Data classes
    planet = Planet()

    t_cut, f_cut = t_full[ind], f_full[ind]
    data = Data("01_lc.dat", cut=ind)

    # parameters for emcee
    # ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, planet.N_free_parameters*3, 100, 40      # testing
    ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 60, 500, 250       # basic
    # ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 300, 2000, 1000    # fitting

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
    plt.savefig(out_dir + str(count) + "_samples_all.pdf", format="pdf")
    # plt.show()
    plt.close()

    # get the medians of the posterior distributions
    median_pars = np.median(samples, axis=0)

    # print(['%9s' % s.replace('$', '').replace('_', '').replace('\\rm', '').replace('\\', '').replace(' d', '')
    #        for s in planet.labels[:4]])
    print count, "-", np.asarray(['%9.4f' % s for s in median_pars])
    pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))

    save_str = ""
    for pc in pcs:
        # vals = list(pc)
        # save_str += (",".join(vals) + "\n")
        save_str += (str(pc) + "\n")
    with open(out_dir + str(count) + "_fit_vals.dat", "w") as save_file:
        save_file.write(save_str)

    planet.set_transit_parameters(median_pars)
    full_model = planet.get_transit_curve(data.LCtime)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    period, t0, rp, inc, pars = planet.period_01, planet.t0_01, planet.rp_01, planet.inc_01, median_pars

    a = keplerslaw(period)
    b = calc_b(a, inc)
    rp = rp * st_r * myc.RS / myc.RE

    t = np.linspace(data.LCtime.min(), data.LCtime.max(), data.LCtime.size*100)     # supersample
    lc = single_model(pars, t)

    # in_transit_mask = single_model(pars, data.LCtime) < 1.
    # with open("in_transit_mask.dat", "w") as df:
    #     for i in range(len(in_transit_mask)):
    #         df.write("{}\n".format(in_transit_mask[i]))
    # plt.plot(data.LCtime[in_transit_mask], data.LC[in_transit_mask], 'o', ms=3, alpha=0.7, zorder=0, c="r")
    # plt.plot(data.LCtime[in_transit_mask==0], data.LC[in_transit_mask==0], 'o', ms=3, alpha=0.7, zorder=1)
    # plt.plot(t, lc, lw=2, alpha=0.8, zorder=2)
    # plt.xlim([min(data.LCtime), max(data.LCtime)])
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3, 1]})

    # plot full LC and single model
    ax1.plot(data.LCtime, data.LC, 'o', ms=3, alpha=0.7, zorder=1)
    ax1.plot(t_cut, f_cut, '*', ms=10, alpha=0.7, zorder=1, c="r")
    ax1.plot(t, lc, lw=2, alpha=0.8, zorder=2)
    ax1.set_xlim([min(data.LCtime), max(data.LCtime)])

    # plot phase-folded LC and ss-single model
    phase, lcfold = phase_fold(data.LCtime, data.LC, period, t0)
    p_cut = ((((t_cut-t0+period/2.0) / period) % 1) - 0.5) * period * 24.
    phase = np.asarray(phase)
    phase = (phase - 0.5) * period * 24.    # hours from transit centre
    ax2.plot(phase, lcfold, 'o', ms=3, alpha=0.7, zorder=1)
    ax2.plot(p_cut, f_cut, '*', ms=10, alpha=0.7, zorder=1, c="r")
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
    plt.savefig(out_dir + str(count) + "_fit.pdf", format="pdf")
    # plt.show()
    plt.close()

    with open(out_dir + str(count) + "_mcmc.pkl", "wb") as pklf:
        dill.dump([data, planet, samples], pklf)

    count += 1
