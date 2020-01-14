from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import batman
from lmfit import minimize, Parameters, Parameter, report_fit
import astropy.io.ascii as ascii
import scipy.interpolate as interpolate
from exoplanet import phase_fold
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import sys, os


mam = ascii.read('/Users/rwells/Desktop/K2/mamajek_asciiread.txt', fill_values=[('....', '0.0'), ('...', '0.0')])
Tgrid = mam['Teff']
Mgrid = mam['Msun']
logLgrid = mam['logL']
fM = interpolate.interp1d(Tgrid, Mgrid)
fL = interpolate.interp1d(Tgrid, logLgrid)
sigma = 5.6704e-5       # erg /s / cm62 / K64
Lbolsun = 3.8270e33     # erg/s
Rsun = 6.957e10         # cm
Rjup = 69911000e2       # cm
Rearth = 6371e5         # cm


def calc_R_M_dist(Teff):
    M = fM(Teff)
    logL = fL(Teff)
    # get radius from luminosity:
    Lbol = 10 ** logL * Lbolsun
    R = np.sqrt(Lbol / (4 * np.pi * sigma * Teff ** 4)) / Rsun

    return [R, M]


def keplerslaw_teff(kper, teff):
    st_r, st_m = calc_R_M_dist(teff)
    """
     Function used to relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period.
    """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m
    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def transitmin(params, time, flux, fluxerr, teff, fit_teff):
    """ Calculate the transit light curve and the difference from the observed flux """
    parbat = batman.TransitParams()
    parbat.t0 = params['t0'].value      # time of transit
    parbat.per = params['per'].value    # orbital period
    parbat.rp = params['rp'].value      # planet radius (in units of stellar radii)
    # parbat.a = params['a'].value        # semi-major axis (in units of stellar radii)
    if fit_teff:
        parbat.a = keplerslaw_teff(params['per'].value, params['teff'].value)   # keplers law
    else:
        parbat.a = keplerslaw_teff(params['per'].value, teff)   # keplers law
    parbat.inc = params['inc'].value    # orbital inclination (in degrees)
    parbat.ecc = params['ecc'].value    # eccentricity
    parbat.w = params['w'].value        # longitude of periastron (in degrees)
    parbat.u = [params['u1'].value, params['u2'].value]    # limb darkening coefficients
    parbat.limb_dark = "quadratic"                         # limb darkening model

    m = batman.TransitModel(parbat, time, supersample_factor=21, exp_time=0.5/24.)
    model = m.light_curve(parbat)

    return (model - flux)/fluxerr


def fit_transit(time, lc, lcerror, period, epoch, rp, teff, fit_teff=False):

    # assume limb darkening
    u1 = 0.5079     # 0.4983
    u2 = 0.2239     # 0.2042

    # set up the parameters for batman
    params2 = batman.TransitParams()
    params2.t0 = epoch                  # time of inferior conjunction
    params2.per = period                # orbital period
    params2.rp = rp                     # planet radius (in units of stellar radii)
    params2.a = keplerslaw_teff(period, teff)
    params2.inc = 90.                   # orbital inclination (in degrees)
    params2.ecc = 0.                    # eccentricity
    params2.w = 90.                     # longitude of periastron (in degrees)
    params2.u = [u1, u2]                # limb darkening coefficients
    params2.limb_dark = "quadratic"     # limb darkening model

    # calculate and plot the starting model
    m = batman.TransitModel(params2, time, supersample_factor=21, exp_time=0.5/24.)    # initializes model
    model = m.light_curve(params2)      # initial model

    # create a set of parameters for lmfit - the ones that are fixed have vary=false
    params = Parameters()
    params.add('t0', value=epoch, min=epoch-period/2., max=epoch+period/2., vary=True)
    params.add('per', value=period, vary=True)
    params.add('rp', value=rp, min=0., max=1.0)
    # params.add('a', value=17.0, min=1.)
    params.add('inc', value=90., vary=True, min=70., max=90.)
    params.add('ecc', value=0.0, vary=False)
    params.add('w', value=90., vary=False)
    params.add('u1', value=u1, vary=False, min=0., max=1.0)
    params.add('u2', value=u2, vary=False, min=0., max=1.0)
    if fit_teff:
        params.add('teff', value=teff, vary=True, min=2000., max=7500.)

    # use lmfit to estimate the transit parameters
    result = minimize(transitmin, params, args=(time, lc, lcerror, teff, fit_teff), method="leastsq")

    # report_fit(result)
    # print("               ", [key for key in params.keys()])
    # print("Priors        =", [("%.4f" % params[p].value) for p in params.keys()])
    # print("Fitted values =", [("%.4f" % result.params[p].value) for p in result.params.keys()])

    # recover the fitted parameters from the result
    parbat = batman.TransitParams()
    parbat.t0 = result.params['t0'].value       # time of inferior conjunction
    parbat.per = result.params['per'].value     # orbital period
    parbat.rp = result.params['rp'].value       # planet radius (in units of stellar radii)
    # parbat.a = result.params['a'].value         #semi-major axis (in units of stellar radii)
    if fit_teff:
        parbat.a = keplerslaw_teff(result.params['per'].value, result.params['teff'].value)
    else:
        parbat.a = keplerslaw_teff(result.params['per'].value, teff)
    parbat.inc = result.params['inc'].value     # orbital inclination (in degrees)
    parbat.ecc = 0.                             # eccentricity
    parbat.w = 90.                              # longitude of periastron (in degrees)
    parbat.u = [u1, u2]                         # limb darkening coefficients
    parbat.limb_dark = "quadratic"              # limb darkening model

    period = result.params['per'].value
    epoch0 = result.params['t0'].value
    if fit_teff:
        tstar = result.params['teff'].value
    else:
        tstar = teff
    rs, ms = calc_R_M_dist(tstar)
    rj = result.params['rp'].value * rs * Rsun / Rjup
    re = result.params['rp'].value * rs * Rsun / Rearth

    # print(int(tstar), round(rs, 2), round(ms, 2), round(rj, 1), round(re, 1))
    # print(parbat.a)

    timefold, fluxfold = phase_fold(time, lc, period, epoch0)

    m = batman.TransitModel(parbat, time, supersample_factor=21, exp_time=0.5/24.)
    tmodel = m.light_curve(parbat)
    _, fitmodel = phase_fold(time, tmodel, period, epoch0)
    timefold, fitmodel = np.asarray(timefold), np.asarray(fitmodel)

    pnames = ['t0', 'per', 'rp', 'a', 'inc', 'teff']
    vals, errs = {}, {}
    for p in pnames:
        if p in result.var_names:
            vals[p] = result.params[p].value
            errs[p] = result.params[p].stderr
        elif p == 'a':
            vals[p] = parbat.a
            errs[p] = 0.5
        elif p == 'teff':
            vals[p] = teff
            errs[p] = 0.0

    # s = ["%.5f|%.5f" % (vals[p], errs[p]) for p in pnames]
    # print(",".join(s))

    # if plot:
    #     plt.figure(figsize=(12, 7))
    #     plt.plot(timefold, fluxfold, '.')
    #     plt.plot(timefold, fitmodel, color='r')
    #     plt.xlim(0, 1)
    #     plt.savefig("plots/c{}/{}_fit_phase_fold_full.pdf".format(c, k2id))
    #     plt.close("all")
    #
    #     timefold = (timefold-0.25) * period * 24.
    #     pr *= (period * 24.)
    #     plt.figure(figsize=(10, 7))
    #     plt.plot(timefold, fluxfold, ".")
    #     plt.plot(timefold, fitmodel, color="r")
    #     plt.xlim(-pr, pr)
    #     plt.savefig("plots/c{}/{}_fit_phase_fold_zoom.pdf".format(c, k2id))
    #     plt.close("all")

    # ## cut the transits and write them to a file
    # filename = 'cuttransits.txt'
    # fp = open(filename, 'w')
    # for i in np.where(abs(phase) < phadur)[0]:
    #     fp.write('%.10f\t%f\t%f\n'%(time[i], lc[i], lcerror[i]))
    # fp.close()

    return tmodel, timefold, fluxfold, fitmodel, vals, rj, re, result.redchi, result.bic, result.nfev


def bls_search(time, flux, err, per_range=(1., 80.), q_range=(0.001, 0.115), nf=10000, nbin=900, plot=False, k2id=None,
               n_i=1, path="BLS/", mf=201):
    from pybls import BLS
    bls = BLS(time, flux, err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()  # do BLS search

    sde_trans, p_ar, p_pow, bper, t0, rprs, depth, p_sde = \
        res.bsde, bls.period, res.p, res.bper, bls.tc, np.sqrt(res.depth), res.depth, res.sde

    test_pow = p_pow #- medfilt(p_pow, mf)  # running median
    # test_pow = p_pow - median_filter(p_pow, mf)
    p_sde = test_pow / test_pow.std()  # SDE

    # p_sde = (p_pow - p_pow.mean()) / p_pow.std()

    sde_trans = np.nanmax(p_sde)
    bper = p_ar[(np.abs(p_sde - sde_trans)).argmin()]

    pr = min([(res.in2 - res.in1) / 24. / bper, 0.5])   # limit plot to 2 durations in phase

    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(p_ar, p_sde, lw=2)
        plt.xlim([min(p_ar), max(p_ar)])
        plt.ylim([0., int(sde_trans) + 1])
        # plt.savefig("{}{}_BLS_{}.pdf".format(path, k2id, n_i))
        # plt.close("all")
        plt.show()

        p, f = phase_fold(time, flux, bper, t0)
        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax.plot(p, f, ".")
        ax.set_xlim(0.5-pr, 0.5+pr)
        # plt.savefig("{}{}_BLS_{}_phase_fold.pdf".format(path, k2id, n_i))
        # plt.close("all")
        plt.show()

    return sde_trans, bper, t0, rprs, pr


def fit_transit_pyastro(t, f, e, bper, t0, rp, teff, fit_teff=False):
    from PyAstronomy.modelSuite import forTrans as ft
    st_r, st_m = calc_R_M_dist(teff)

    def getRelation():
        def keplerslaw(kper):
            """
             Function used to relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period.
            """

            Mstar = st_m * 1.989e30  # kg
            G = 6.67408e-11  # m3 kg-1 s-2
            Rstar = st_r * 695700000.  # m
            return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar

        return keplerslaw

    ma = ft.MandelAgolLC(orbit="circular", ld="quad")  # initialise model

    # set guesses for MCMC
    ma["per"] = bper    # period
    ma["i"] = 90.0      # inclination
    ma["a"] = 0.1       # semi-major axis in stellar radii (dummy value - linked to period)
    ma["T0"] = t0       # time of first transit
    ma["p"] = rp        # planet-star radius ratio
    ma["linLimb"] = 0.5079      # limb darkening coefficient
    ma["quadLimb"] = 0.2239     # limb darkening coefficient
    ma["b"] = 0.0       # planet-star flux ratio (not impact parameter!)

    ma.thaw(["T0", "p", "per", "a", "i"])  # free parameters for fit
    ma.relate("a", ["per"], getRelation())

    X0, lims, steps = ma.MCMCautoParameters(
        {"per": [ma['per'] - 0.2, ma['per'] + 0.2], "T0": [ma['T0'] - 0.2, ma['T0'] + 0.2],
         "p": [0.01, 0.2], "i": [70.0, 90.0]}, picky=True)

    sys.stdout = open(os.devnull, 'w')  # suppress printing
    ma.fitMCMC(t, f, X0, lims, steps, yerr=e, iter=2000, burn=500, quiet=True)
    sys.stdout = sys.__stdout__     # enable printing

    ma.updateModel()
    tmodel = ma.model  # output model (vs t)

    rj = ma['p'] * st_r * Rsun / Rjup
    re = ma['p'] * st_r * Rsun / Rearth

    timefold, fluxfold = phase_fold(t, f, ma["per"], ma["T0"])

    _, fitmodel = phase_fold(t, tmodel, ma["per"], ma["T0"])
    timefold, fitmodel = np.asarray(timefold), np.asarray(fitmodel)

    pnames = ['t0', 'per', 'rp', 'a', 'inc', 'teff']
    vals = {}
    for p, ma_p in zip(pnames, ["T0", "per", "p", "a", "i", "teff"]):
        if p == 'teff':
            vals[p] = teff
        else:
            vals[p] = ma[ma_p]

    return tmodel, timefold, fluxfold, fitmodel, vals, rj, re
