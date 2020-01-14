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
import pandas
import corner
import dill
from pathos.multiprocessing import Pool
from tqdm import tqdm
from time import time

Rsun = myc.RS
Rjup = myc.RJ

st_r, st_m = 0.455, 0.461


def calc_i(_a, _b):
    return np.degrees(np.arccos(_b / _a))


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


# TODO
# remove other transits from LC
# relax priors
# plot each transit fit
# save samples to pkl
# run on kelvin - extend steps
# use same amount of points


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

        global rprs, inc, per
        # free parameters
        self.rp = rprs          # planet radius (in units of stellar radii)
        self.inc = inc         # orbital inclination (in degrees)
        self.a = keplerslaw(per, self.stellar_radius, self.stellar_mass)
        self.period = per      # orbital period

        self.t0 = 0.          # time of inferior conjunction
        self.N_free_parameters = 1     # number of free parameters in model

        # start the batman model
        self.params = batman.TransitParams()
        self.params.limb_dark = self.limb_dark  # limb darkening model
        self.params.u = self.u                  # limb darkening coefficients

        # names of the parameters for the corner plot
        # self.labels = [r'$R_p/R_*$', r'$i$', r'$T_0$', r'$P$']
        # if fit_ecc:
        #     self.labels += [r'$e$', r'$w$']

    def set_transit_parameters(self, pars):
        """ set transit parameters from pars """
        self.t0 = pars[0]
        # if fit_ecc:
        #     self.ecc, self.w = pars[-2:]
        # self.a = keplerslaw(self.period, self.stellar_radius, self.stellar_mass)
    
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
        global t01, t02
        """ define each parameter prior distribution """
        # rp1,rp2 = [float(x) for x in row.prior_rp.split("|")]
        # i1,i2 = [float(x) for x in row.prior_inc.split("|")]
        # t01,t02 = [float(x) for x in row.prior_t0.split("|")]
        # p1,p2 = [float(x) for x in row.prior_period.split("|")]
        # e1, e2 = [float(x) for x in row.prior_ecc.split("|")]
        # w1, w2 = [float(x) for x in row.prior_w.split("|")]
        # self.prior_rp = stats.norm(rp1, rp2)    # planet radius (in units of stellar radii)
        # self.prior_inc = stats.uniform(i1, i2)  # orbital inclination (in degrees)
        self.prior_t0 = stats.norm(t01, t02)    # time of inferior conjunction (in BJD)
        # self.prior_period = stats.norm(p1, p2)  # orbital period (in days)
        # if fit_ecc:
        #     self.prior_ecc = stats.uniform(e1, e2)  # eccentricity
        #     self.prior_w = stats.uniform(w1, w2)

    def get_from_prior(self, nwalkers):
        """ return a list with random values from each parameter's prior """
        self.set_priors()   # use distributions from set_priors

        # if fit_ecc:
        #     pfp += [self.prior_ecc.rvs(nwalkers), self.prior_w.rvs(nwalkers)]
        pars_from_prior = self.prior_t0.rvs(nwalkers)

        # with open("pkl.pkl", "wb") as pkf:
        #     dill.dump(pars_from_prior, pkf)
        # sys.exit()

        return np.asarray([pars_from_prior]).T


class Data(object):
    """ GLOBAL class to hold the light curve """
    def __init__(self, lc_file, skip_lc_rows=0, m1=None, m2=None):
        self.lc_file = lc_file

        # read light curve
        self.LCtime, self.LC, self.LCerror = np.loadtxt(lc_file, unpack=True, skiprows=skip_lc_rows)

        mask = ((self.LCtime > m1 - m2) & (self.LCtime < m1 + m2))
        self.LCtime, self.LC, self.LCerror = self.LCtime[mask], self.LC[mask], self.LCerror[mask]

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
    # elif fit_ecc and (pars[-2] > e2 or pars[-1] > w2):
    #     return -np.inf
    else:
        return log_like


def lnprior(pars, planet):
    """ calculate the log prior for a set of parameters """
    # transit parameters
    # prior_rp = planet.prior_rp.logpdf(pars[0])
    # prior_inc = planet.prior_inc.logpdf(pars[1])
    prior_t0 = planet.prior_t0.logpdf(pars[0])
    # prior_period = planet.prior_period.logpdf(pars[3])

    # ln_prior = prior_rp + prior_inc + prior_t0 + prior_period
    # if fit_ecc:
    #     prior_ecc = planet.prior_ecc.logpdf(pars[-2])
    #     prior_w = planet.prior_w.logpdf(pars[-1])
    #     ln_prior += prior_ecc
    #     ln_prior += prior_w

    return prior_t0


# ofile = open("probs.dat", "wa")

def lnprob(pars, planet):
    """ posterior distribution """
    log_prior = lnprior(pars, planet)
    log_like = lnlike(pars, planet)
    log_posterior = log_prior + log_like

    # all_lnprobs.append([list(pars), log_prior, log_like])
    # ofile.write(str([list(pars), log_prior, log_like]) + "\n")

    return log_posterior


run = "save_1_16_150_1600_800"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    fitted_data, fitted_planet, fitted_samples = dill.load(pklf)

all_samples = [[], [], [], []]
all_data = [[], [], [], []]
all_planet = [[], [], [], []]
for pl in range(4):
    rprs = np.median(fitted_samples.T[0 + pl*4])
    inc = np.median(fitted_samples.T[1 + pl*4])
    t0 = np.median(fitted_samples.T[2 + pl*4])
    t0_err = np.average(np.abs(np.percentile(fitted_samples.T[2 + pl*4], [50., 5., 95.])[1:] -
                               np.percentile(fitted_samples.T[2 + pl*4], [50., 5., 95.])[0]))
    per = np.median(fitted_samples.T[3 + pl*4])
    per_err = np.average(np.abs(np.percentile(fitted_samples.T[3 + pl*4], [50., 5., 95.])[1:] -
                                np.percentile(fitted_samples.T[3 + pl*4], [50., 5., 95.])[0]))
    n_trans = int(np.floor((fitted_data.LCtime[-1] - t0) / per) + 1)
    for tr in range(n_trans):
        # initialize the Planet and Data classes
        t01, t02 = t0 + per*tr, t0_err + per_err*tr
        planet = Planet()
        data = Data(lc_file="final-lc-mySFF.dat", m1=t0+tr*per, m2=5./24.)

        # parameters for emcee
        ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 100, 500, 300       # testing
        # ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 100, 1000, 500    # basic
        # ndim, nwalkers, nsteps, burnin = planet.N_free_parameters, 500, 5000, 4000   # fitting

        # get random starting positions from the priors
        pos = planet.get_from_prior(nwalkers)
        # print "Priors between", [(min(pri), max(pri)) for pri in np.asarray(pos).T]

        # This now uses pathos !!
        pool = Pool(processes=1)   # use 4 cores for ~2 times speed
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(planet,), pool=pool)
        out = sampler.run_mcmc(pos, nsteps, progress=True)

        # remove burn-in
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        planet.set_transit_parameters([np.median(np.asarray(samples).flatten())])

        all_samples[pl].append(samples)
        all_data[pl].append(data)
        all_planet[pl].append(planet)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.errorbar(data.LCtime, data.LC, yerr=data.LCerror,
                    fmt='--o', alpha=0.7, zorder=1, color="k", ms=10, elinewidth=1.5)
        ax.plot(data.LCtime, planet.get_transit_curve(data.LCtime), lw=2, alpha=0.8, zorder=2)
        # ax1.set_xlim([min(data.LCtime), max(data.LCtime)])

        plt.tight_layout()
        plt.savefig("ttv-plots/{pl}-{tr}.pdf".format(**locals()), format="pdf")
        plt.close("all")

with open("ttv-mcmc.pkl", "wb") as pklf:
    dill.dump([all_data, all_planet, all_samples], pklf)
