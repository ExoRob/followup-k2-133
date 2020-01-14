import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import emcee
from tqdm import tqdm
import sys
import pandas
import bayesflare.models.model as m
import corner
import dill


# uniform priors
tp1, tp2 = 2., 4.         # peak-flare time
amp1, amp2 = 0.0001, 0.08    # flare amplitude
tg1, tg2 = 0.001, 0.04       # gaussian rise timescale
te1, te2 = 0.1, 0.8        # exponential decay timescale

two_flares = True
tp21, tp22 = 2., 4.         # peak-flare time
amp21, amp22 = 0.0001, 0.08    # flare amplitude
tg21, tg22 = 0.001, 0.04       # gaussian rise timescale
te21, te22 = 0.1, 0.8        # exponential decay timescale

st_l = 0.033    # stellar luminosity in L_Sun


class Flare(object):
    def __init__(self):
        # stellar parameters
        self.stellar_luminosity = st_l

        # free parameters
        self.tpeak = 0.                 #
        self.amplitude = 0.             #
        self.tgaus = 0.                 #
        self.texp = 0.                  #
        self.tpeak2 = 0.  #
        self.amplitude2 = 0.  #
        self.tgaus2 = 0.  #
        self.texp2 = 0.  #

        self.N_free_parameters = 4 * (1 + two_flares)      # number of free parameters in model

        # model parameters
        self.nss = 100
        self.prior_tpeak = 0.
        self.prior_amplitude = 0.
        self.prior_tgaus = 0.
        self.prior_texp = 0.
        self.prior_tpeak2 = 0.
        self.prior_amplitude2 = 0.
        self.prior_tgaus2 = 0.
        self.prior_texp2 = 0.

        # names of the parameters for the corner plot
        self.labels = [r'$t_{peak}$', r'$A$', r'$\tau_{rise}$', r'$\tau_{decay}$']
        if two_flares:
            self.labels += [r'$t2_{peak}$', r'$A2$', r'$\tau2_{rise}$', r'$\tau2_{decay}$']

    def set_parameters(self, pars):
        """ set parameters from pars """
        self.tpeak, self.amplitude, self.tgaus, self.texp = pars[:4]
        if two_flares:
            self.tpeak2, self.amplitude2, self.tgaus2, self.texp2 = pars[4:]

    def get_model(self, t):
        """ return the flare model at times t """
        # set self.params for batman model
        tspace = t[1] - t[0]
        t_ss = np.linspace(t[0]-tspace/2., t[-1]+tspace/2., t.size*self.nss)
        model_ss = m.Flare(ts=t_ss, t0=self.tpeak, amp=self.amplitude).model(pdict={'t0': self.tpeak,
                                                                                    'taugauss': self.tgaus,
                                                                                    'tauexp': self.texp,
                                                                                    'amp': self.amplitude})
        if two_flares:
            model_ss += m.Flare(ts=t_ss, t0=self.tpeak2, amp=self.amplitude2).model(pdict={'t0': self.tpeak2,
                                                                                           'taugauss': self.tgaus2,
                                                                                           'tauexp': self.texp2,
                                                                                           'amp': self.amplitude2})

        model = model_ss.reshape(t.size, self.nss).sum(axis=1) / float(self.nss) + 1.
        # TODO: check normalisation
        # TODO: split models for plotting
        
        return model

    def set_priors(self):
        """ define each parameter prior distribution """
        self.prior_tpeak = stats.uniform(tp1, tp2)        #
        self.prior_amplitude = stats.uniform(amp1, amp2)  #
        self.prior_tgaus = stats.uniform(tg1, tg2)        #
        self.prior_texp = stats.uniform(te1, te2)         #

        if two_flares:
            self.prior_tpeak2 = stats.uniform(tp21, tp22)         #
            self.prior_amplitude2 = stats.uniform(amp21, amp22)   #
            self.prior_tgaus2 = stats.uniform(tg21, tg22)         #
            self.prior_texp2 = stats.uniform(te21, te22)          #

    def get_from_prior(self, nwalkers):
        """ return a list with random values from each parameter's prior """
        self.set_priors()   # use distributions from set_priors

        pfp = [self.prior_tpeak.rvs(nwalkers), self.prior_amplitude.rvs(nwalkers),
               self.prior_tgaus.rvs(nwalkers), self.prior_texp.rvs(nwalkers)]

        if two_flares:
            pfp += [self.prior_tpeak2.rvs(nwalkers), self.prior_amplitude2.rvs(nwalkers),
                    self.prior_tgaus2.rvs(nwalkers), self.prior_texp2.rvs(nwalkers)]

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

        mask = (self.LCtime > 3018.2) & (self.LCtime < 3018.6)
        self.LCtime, self.LC, self.LCerror = self.LCtime[mask], self.LC[mask], self.LCerror[mask]
        self.LCtime = (self.LCtime - self.LCtime[0]) * 24.      # hours

        self.N_lc = self.LCtime.size


def lnlike(pars, flare):
    """ log likelihood function """
    log2pi = np.log(2.0*np.pi)

    # set the transit params
    flare.set_parameters(pars)

    # calculate the lnlike for flare
    flare_model = flare.get_model(data.LCtime)
    sigma = data.LCerror**2.
    chi = np.log(sigma)/2. + (data.LC - flare_model)**2. / (2.*sigma)
    
    log_like_flare = - 0.5*data.N_lc*log2pi - np.sum(chi)

    if (not np.isfinite(log_like_flare)) or (pars[0] > pars[4]) or (pars[4] - pars[0]) > 0.4:
        return -np.inf
    else:
        return log_like_flare


def lnprior(pars, flare):
    """ calculate the log prior for a set of parameters """
    # flare parameters
    prior_tpeak = flare.prior_tpeak.logpdf(pars[0])
    prior_amplitude = flare.prior_amplitude.logpdf(pars[1])
    prior_tgaus = flare.prior_tgaus.logpdf(pars[2])
    prior_texp = flare.prior_texp.logpdf(pars[3])
    if two_flares:
        prior_tpeak2 = flare.prior_tpeak.logpdf(pars[4])
        prior_amplitude2 = flare.prior_amplitude.logpdf(pars[5])
        prior_tgaus2 = flare.prior_tgaus.logpdf(pars[6])
        prior_texp2 = flare.prior_texp.logpdf(pars[7])

    ln_prior = prior_tpeak + prior_amplitude + prior_tgaus + prior_texp
    if two_flares:
        ln_prior += prior_tpeak2 + prior_amplitude2 + prior_tgaus2 + prior_texp2

    return ln_prior


# ofile = open("probs.dat", "wa")

def lnprob(pars, flare):
    """ posterior distribution """
    log_prior = lnprior(pars, flare)
    log_like = lnlike(pars, flare)
    log_posterior = log_prior + log_like

    # all_lnprobs.append([list(pars), log_prior, log_like])
    # ofile.write(str([list(pars), log_prior, log_like]) + "\n")

    if np.isfinite(log_posterior):
        return log_posterior
    else:
        return -np.inf


# initialize the flare and Data classes
flare = Flare()
data = Data(lc_file="lc-none-mySFF.dat")

# parameters for emcee
# ndim, nwalkers, nsteps, burnin = flare.N_free_parameters, 50, 150, 60       # testing
ndim, nwalkers, nsteps, burnin = flare.N_free_parameters, 200, 2000, 1500     # basic
# ndim, nwalkers, nsteps, burnin = flare.N_free_parameters, 300, 4000, 3000   # fitting

# get random starting positions from the priors
pos = flare.get_from_prior(nwalkers)
# print "Priors between", [(min(pri), max(pri)) for pri in np.asarray(pos).T]

# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(flare,))
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
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(flare,))
out = sampler.run_mcmc(pos, nsteps, progress=True)

with open("flare-posprob.pkl", "wb") as pklf:
    dill.dump(out, pklf)

# pos_all = []
# prob_all = []
# for posi, lnp, state in tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
#     pos_all.append(posi[np.argmax(lnp), :])
#     prob_all.append(np.max(lnp))
# with open("flare-posprob.pkl", "wb") as pklf:
#     dill.dump([pos_all, prob_all], pklf)

# remove burn-in
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# make a corner plot
corner.corner(samples, labels=flare.labels)
plt.savefig("flare-samples.pdf", format="pdf")
# plt.show()
plt.close("all")

# get the medians of the posterior distributions
median_pars = np.median(samples, axis=0)

print ['%9s' % s.replace('$', '').replace('_', '').replace('\\rm', '').replace('\\', '') for s in flare.labels]
print ['%9.4f' % s for s in median_pars]
pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))
print pcs

flare.set_parameters(median_pars)

# TODO: peak & total energy [erg]

t = np.linspace(data.LCtime.min(), data.LCtime.max(), data.LCtime.size*100)
lc_ss = flare.get_model(t)
lc = flare.get_model(data.LCtime)

fig, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 2]})
ax2.plot(data.LCtime, data.LC, 'o', ms=3, alpha=0.7, zorder=1)
ax2.plot(data.LCtime, lc, lw=2, alpha=0.8, zorder=2)
ax2.plot(t, lc_ss, lw=2, alpha=0.8, zorder=2)
ax2.tick_params(axis='x', labelbottom=False)

resid = data.LC - lc
ax3.plot(data.LCtime, resid, 'o', alpha=0.7, ms=3, zorder=1)
ax3.axhline(0., color='k', alpha=0.7, lw=2, zorder=2)
# _ = [ax3.axhspan(-data.LCerror[0]*i, data.LCerror[0]*i, color="grey", alpha=0.25) for i in np.arange(1, 4)]
# ax3.set_xlim([0.495, 0.505])
# ax3.set_ylim([-data.LCerror[0]*3, data.LCerror[0]*3])

ax2.get_shared_x_axes().join(ax2, ax3)
plt.tight_layout()
plt.savefig("flare-fit.pdf", format="pdf")
# plt.show()
plt.close("all")

with open("flare-mcmc.pkl", "wb") as pklf:
    dill.dump([data, flare, samples], pklf)
