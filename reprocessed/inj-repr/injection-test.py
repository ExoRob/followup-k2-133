import platform
on_kelvin = 'kelvin' in platform.node()
if on_kelvin:
    print "> Running on Kelvin ..."
    import matplotlib
    matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt
import batman
from pybls import BLS
from scipy.ndimage import median_filter
from tqdm import tqdm
import multiprocessing
import dill
import os
import sys
import fit_transit
import seaborn as sns
sns.set()


def keplerslaw(kper):
    """
    Kepler's Law
    :param kper: orbital period in days
    :return: star-planet distance in stellar radii
    """
    st_r, st_m = 0.456, 0.497
    Mstar = st_m * 1.989e30     # kg
    G = 6.67408e-11             # m3 kg-1 s-2
    Rstar = st_r * 695700000.   # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def calc_i(_a, _b):
    """
    Convert impact parameter to inclination
    :param _a: star-planet distance in stellar radii
    :param _b: impact parameter [0, 1]
    :return: inclination in degrees
    """
    return np.degrees(np.arccos(_b / _a))


def batman_to_string(params):
    """
    Convert transit parameters to a string
    :param params: Batman params object
    :return: string of transit parameters
    """
    s = "\nT0   = {:9.4f}\nPer  = {:9.4f}\nRpRs = {:9.4f}\na    = {:9.4f}\nInc  = {:9.4f}\n".\
        format(params.t0, params.per, params.rp, params.a, params.inc)

    return s


def random_transit_model(time, per_lims, rp_lims, depth_lims, n_trans_lims):
    """
    Produce random transit model inside limits
    :param time: LC time in days
    :param per_lims: limits on orbital period in days [len 2]
    :param rp_lims: limits on planet-star radius ratio SQUARED! [len 2]
    :param depth_lims: limits on transit depth [len 2]
    :param n_trans_lims: limits on number of transits [len 2]
    :return: transit model matching time
    """
    iter_count = 0
    in_limits = False
    while not in_limits:    # TODO: speed-up / improve
        phase = np.random.uniform(0., 1.)
        impact = np.random.uniform(0., 1.)

        params = batman.TransitParams()
        params.per = np.random.uniform(per_lims[0], per_lims[1])
        params.t0 = time[0] + phase*params.per
        params.rp = np.sqrt(np.random.uniform(rp_lims[0], rp_lims[1]))
        params.a = keplerslaw(params.per)
        params.inc = calc_i(params.a, impact)
        params.ecc = 0.
        params.w = 90.
        params.u = [0.5079, 0.2239]
        params.limb_dark = "quadratic"

        bat = batman.TransitModel(params, time, supersample_factor=15, exp_time=29.4/60./24.)
        model = bat.light_curve(params)

        depth = 1. - batman.TransitModel(params, np.array([params.t0]),
                                         supersample_factor=15, exp_time=29.4/60./24.).light_curve(params)[0]

        n_trans = int(np.floor((time.max() - params.t0) / params.per)) + 1

        n_dp = sum(model < 1.)

        in_limits = (depth_lims[0] <= depth <= depth_lims[1]) & (n_trans_lims[0] <= n_trans <= n_trans_lims[1])

        iter_count += 1

    # print iter_count, int(depth*1e6), batman_to_string(params)

    return model, params, depth*1e6, n_trans, n_dp, phase, impact


def specific_transit_model(time, t0, per, rp, inc):
    """
    Produce a transit model
    """
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = keplerslaw(params.per)
    params.inc = inc
    params.ecc = 0.
    params.w = 90.
    params.u = [0.5079, 0.2239]
    params.limb_dark = "quadratic"

    bat = batman.TransitModel(params, time, supersample_factor=15, exp_time=29.4/60./24.)
    model = bat.light_curve(params)

    return model, params


def do_bls(time, flux, flux_err, per_range, q_range, nf, inj_pars, nbin=None, binsize=1501, plot=False):
    """
    Run BLS algorithm
    :param time: LC time in days
    :param flux: LC flux normalised to 1
    :param flux_err: LC error [len time]
    :param per_range: range of periods to search [len 2]
    :param q_range: range of duration-period ratios to search [len 2]
    :param nf: number of periods to search [int]
    :param inj_pars: injected transit params
    :param nbin: number of bins to use in phase-fold [int, None]
    :param binsize: window of median filter [int]
    :param plot: plot the BLS spectrum [bool]
    :return: highest SDE and corresponding period
    """
    if not nbin:
        nbin = time.size

    bls = BLS(time, flux, flux_err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()

    p_ar, p_pow = bls.period, res.p     # power at each period
    rmed_pow = p_pow - median_filter(p_pow, binsize)
    p_sde = rmed_pow / rmed_pow.std()   # SDE

    sde_trans = np.nanmax(p_sde)        # highest SDE
    bper = p_ar[np.argmax(p_sde)]       # corresponding period
    t0 = bls.tc
    rp = np.sqrt(res.depth)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(p_ar, p_sde, lw=2, zorder=3)
        plt.plot(bper, sde_trans, "*", ms=20, label="BLS period", zorder=5)
        plt.axvline(inj_pars.per, label="Injected period", c="k", ls="-", lw=15, alpha=0.2)
        plt.xlim(0, p_ar[0])
        plt.ylim(0, int(sde_trans)+1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sde_trans, bper, t0, rp


t, f, e = np.loadtxt("lc-none-mySFF.dat", unpack=True, dtype=float)
per_range = (1., (t[-1] - t[0]) / 2.)
q_range = (1./24./per_range[1], 5./24.)
nf = 50000   # 50000
bs = 1501    # 1501


def injection_test(ind):
    np.random.seed(None)    # for truly random numbers
    m, pars, depth, ntrans, ndp, injphase, impact = \
        random_transit_model(t, per_range, (.7e-4, 6e-3), (200e-6, 1500e-6), (3, 100))

    f_i = f + m - 1.    # injected flux

    # plt.plot(t, f_i, "k.", ms=3)
    # plt.plot(t, m)
    # plt.show()

    sde, period, t0, rp = do_bls(t, f_i, e, per_range, q_range, nf, pars, binsize=bs, plot=False)

    vals, errs = fit_transit.fit_transit(t, f, e, period, t0, rp, 0.455, 0.481)
    t0, period, rp, a, b = vals['t0'], vals['per'], vals['rp'], vals['a'], vals['b']
    phase = (t0 - t[0]) / period % 1.

    if (sde >= sde_lim) and \
            (pars.per*(1.-tol) <= period <= pars.per*(1.+tol)) and \
            ((injphase-tol) <= phase <= (injphase+tol)):
        recovered = 1
    else:
        recovered = 0

    # print "Phase {:.4f} | {:.4f}\nPeriod {:.4f} | {:.4f}\nSDE {:.4f}\nRecovered {}\n".\
    #     format(injphase, phase, pars.per, period, sde, bool(recovered))

    return ind, recovered, depth, sde, ndp, pars, ntrans, injphase, impact


n_tests = 3000     # number of injections to do
tol = 0.01          # tolerance on period and epoch in fraction (percent/100)
sde_lim = 8.        # minimum SDE to detect
ncores = 20         # number of processes to run concurrently


# print "> Running injection tests ..."
# for i in range(n_tests):
#     ind, recovered, depth, sde, ndp, pars, ntrans, injphase, impact = injection_test(i)


rec_ar, depth_ar, sde_ar, dp_ar, pars_ar, ntrans_ar, phase_ar, impact_ar = \
    np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], int), np.array([]), \
    np.array([], int), np.array([]), np.array([])

pool = multiprocessing.Pool(processes=ncores)
for res in tqdm(pool.imap_unordered(injection_test, range(n_tests)), total=n_tests, desc="Running injection tests"):
    rec_ar = np.append(rec_ar, int(res[1]))
    depth_ar = np.append(depth_ar, int(res[2]))
    sde_ar = np.append(sde_ar, float(res[3]))
    dp_ar = np.append(dp_ar, int(res[4]))
    pars_ar = np.append(pars_ar, res[5])
    ntrans_ar = np.append(ntrans_ar, res[6])
    phase_ar = np.append(phase_ar, res[7])
    impact_ar = np.append(impact_ar, res[8])

n_rec = rec_ar.sum()
n_not = n_tests - n_rec
depth_rec = depth_ar[rec_ar == 1]
depth_not = depth_ar[rec_ar == 0]

param_dists = [[par.per for par in pars_ar], [par.t0 for par in pars_ar], [par.rp for par in pars_ar],
               [par.inc for par in pars_ar], depth_ar, dp_ar]
param_dists = [np.asarray(d) for d in param_dists]

run = 0
while os.path.exists("inj-res{}.pkl".format(run)):
    run += 1

with open("inj-res{}.pkl".format(run), "wb") as pf:
    dill.dump([rec_ar, depth_ar, sde_ar, pars_ar, dp_ar, ntrans_ar, phase_ar, impact_ar, param_dists], pf)

fig = plt.figure(figsize=(14, 8))
for i in range(6):
    fig.add_subplot(2, 3, i+1)
    dist = param_dists[i]                   # total
    dist_rec = param_dists[i][rec_ar == 1]  # recovered
    dist_not = param_dists[i][rec_ar == 0]  # missed
    if i < 5:
        bins = np.linspace(dist.min(), dist.max(), 40)
    else:
        bins = np.linspace(dist.min(), dist.max(), min([len(set(dist)), 40]))

    plt.hist(dist, bins=bins, color="0.4")
    plt.hist(dist_rec, bins=bins, color="g", alpha=0.5)
    plt.hist(dist_not, bins=bins, color="r", alpha=0.5)
    plt.title(["Per", "T0", "Rp", "Inc", "Depth", "DP"][i])
plt.tight_layout()
plt.savefig("all-dists{}.pdf".format(run))
# plt.show()
plt.close("all")

# print
# print depth_ar
# print sde_ar

print "{:d} were recovered ({:.2f}%)".format(n_rec, float(n_rec)/float(n_rec+n_not)*100.)

sns.regplot(depth_ar, sde_ar)
plt.axvline(1044., color="g", linestyle="-", label="Candidate")
plt.axhline(sde_lim, color="k", linestyle="--", label="SDE limit")
plt.xlabel("Depth (ppm)")
plt.ylabel("SDE")
plt.legend()
plt.tight_layout()
plt.savefig("depth-sde-correlation{}.pdf".format(run))
# plt.show()
plt.close("all")


# TODO: normalise each distribution to their totals
# sns.kdeplot(depth_rec, shade=True, color="g", label="Recovered")
# sns.kdeplot(depth_not, shade=True, color="r", label="Missed")
plt.hist(depth_rec, bins=50, color="g", label="Recovered")
plt.hist(depth_not, bins=50, color="r", label="Missed")
plt.axvline(1044., color="k", linestyle="--", label="Candidate")
plt.xlabel("Depth (ppm)")
plt.legend()
plt.tight_layout()
plt.savefig("injection-test-kde{}.pdf".format(run))
# plt.show()
plt.close("all")
