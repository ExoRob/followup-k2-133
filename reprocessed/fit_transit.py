import numpy as np
import matplotlib.pyplot as plt
import batman
from lmfit import minimize, Parameters, Parameter, report_fit
from exoplanet import phase_fold
from scipy.ndimage import median_filter

Rsun = 6.957e10         # cm
Rjup = 69911000e2       # cm
Rearth = 6371e5         # cm


def calc_i(_a, _b):
    return np.degrees(np.arccos(_b / _a))


def keplerslaw(kper, st_r, st_m):
    """
     Function used to relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period.
    """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m
    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def transitmin(params, time, flux, fluxerr, st_r, st_m):
    """ Calculate the transit light curve and the difference from the observed flux """
    parbat = batman.TransitParams()
    parbat.t0 = params['t0'].value      # time of transit
    parbat.per = params['per'].value    # orbital period
    parbat.rp = params['rp'].value      # planet radius (in units of stellar radii)
    parbat.a = keplerslaw(params['per'].value, st_r, st_m)   # keplers law
    parbat.inc = calc_i(parbat.a, params['b'].value)    # orbital inclination (in degrees)
    parbat.ecc = params['ecc'].value    # eccentricity
    parbat.w = params['w'].value        # longitude of periastron (in degrees)
    parbat.u = [params['u1'].value, params['u2'].value]    # limb darkening coefficients
    parbat.limb_dark = "quadratic"                         # limb darkening model

    m = batman.TransitModel(parbat, time, supersample_factor=15, exp_time=0.5/24.)
    model = m.light_curve(parbat)

    return (model - flux)/fluxerr


def fit_transit(time, lc, lcerror, period, epoch, rp, st_r, st_m, method="leastsq", plot=False):

    # assume limb darkening
    u1 = 0.5079
    u2 = 0.2239

    # create a set of parameters for lmfit - the ones that are fixed have vary=false
    b = 0.3
    for itr in range(1, 11):
        params = Parameters()
        params.add('t0', value=epoch, min=epoch-period/(2.*itr), max=epoch+period/(2.*itr), vary=True)
        params.add('per', value=period, vary=True)
        params.add('rp', value=rp, min=0., max=1.0)
        # params.add('a', value=17.0, min=1.)
        params.add('b', value=b, vary=True, min=0., max=1.)
        params.add('ecc', value=0.0, vary=False)
        params.add('w', value=90., vary=False)
        params.add('u1', value=u1, vary=False, min=0., max=1.0)
        params.add('u2', value=u2, vary=False, min=0., max=1.0)

        # use lmfit to estimate the transit parameters
        result = minimize(transitmin, params, args=(time, lc, lcerror, st_r, st_m), method=method)

        epoch, period, rp, b = result.params['t0'].value, result.params['per'].value, result.params['rp'].value, \
            result.params['b'].value

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
    parbat.a = keplerslaw(result.params['per'].value, st_r, st_m)
    parbat.inc = calc_i(parbat.a, result.params['b'].value)     # orbital inclination (in degrees)
    parbat.ecc = 0.                             # eccentricity
    parbat.w = 90.                              # longitude of periastron (in degrees)
    parbat.u = [u1, u2]                         # limb darkening coefficients
    parbat.limb_dark = "quadratic"              # limb darkening model

    period = result.params['per'].value
    epoch0 = result.params['t0'].value
    re = result.params['rp'].value * st_r * Rsun / Rearth

    # print(int(tstar), round(rs, 2), round(ms, 2), round(rj, 1), round(re, 1))
    # print(parbat.a)

    timefold, fluxfold = phase_fold(time, lc, period, epoch0)
    m = batman.TransitModel(parbat, time, supersample_factor=15, exp_time=0.5/24.)
    tmodel = m.light_curve(parbat)
    _, fitmodel = phase_fold(time, tmodel, period, epoch0)
    timefold, fitmodel = np.asarray(timefold), np.asarray(fitmodel)

    pnames = ['t0', 'per', 'rp', 'a', 'b']
    vals, errs = {}, {}
    for p in pnames:
        if p in result.var_names:
            vals[p] = result.params[p].value
            errs[p] = result.params[p].stderr
        elif p == 'a':
            vals[p] = parbat.a
            errs[p] = 0.

    # s = ["%.5f|%.5f" % (vals[p], errs[p]) for p in pnames]
    # print(",".join(s))

    if plot:
        plt.figure(figsize=(12, 7))
        plt.plot(timefold, fluxfold, '.')
        plt.plot(timefold, fitmodel, color='r')
        plt.xlim(0.465, 0.535)
        plt.show()
        # plt.savefig("plots/c{}/{}_fit_phase_fold_full.pdf".format(c, k2id))
        # plt.close("all")

        # timefold = (timefold-0.25) * period * 24.
        # plt.figure(figsize=(10, 7))
        # plt.plot(timefold, fluxfold, ".")
        # plt.plot(timefold, fitmodel, color="r")
        # plt.show()
        # plt.savefig("plots/c{}/{}_fit_phase_fold_zoom.pdf".format(c, k2id))
        # plt.close("all")

    # ## cut the transits and write them to a file
    # filename = 'cuttransits.txt'
    # fp = open(filename, 'w')
    # for i in np.where(abs(phase) < phadur)[0]:
    #     fp.write('%.10f\t%f\t%f\n'%(time[i], lc[i], lcerror[i]))
    # fp.close()

    return vals, errs


if __name__ == "__main__":
    t, f, e = np.loadtxt("lc-mySFF.dat", unpack=True)

    vals, errs = fit_transit(t, f, e, 25.6, 3004.86, 0.035, 0.455, 0.461, plot=True)
    # vals, errs = fit_transit(t, f, e, 11.02, 2993.17, 0.04, 0.455, 0.461, plot=True)

    for par in vals.keys():
        print par, vals[par], errs[par]
