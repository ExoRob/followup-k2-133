import numpy as np
import matplotlib.pyplot as plt
import batman
from pybls import BLS
from scipy.signal import medfilt
from tqdm import tqdm
import seaborn as sns
sns.set()

t, f, e = np.loadtxt("LC_K2SC_mask_unclipped_notransits.dat", unpack=True, delimiter=",", dtype=float)

mask = (f < 1. + 3.*e[0]) & (f > 1. - 5.*e[0])  # clean outliers

# plt.plot(t[mask], f[mask], ".")
# plt.plot(t[mask==0], f[mask==0], "X", c="r")
# plt.show()
#
# sns.distplot(f[mask]-1., bins=200)
# plt.show()

t, f, e = t[mask], f[mask], e[mask]


def keplerslaw(kper):
    """
    Kepler's Law
    :param kper: orbital period in days
    :return: star-planet distance in stellar radii
    """
    st_r, st_m = 0.456, 0.497
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

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


def random_transit_model(time, t0_lims, per_lims, rp_lims, depth_lims, n_trans):
    """
    Produce random transit model inside limits
    :param time: LC time in days
    :param t0_lims: limits on transit epoch [len 2]
    :param per_lims: limits on orbital period in days [len 2]
    :param rp_lims: limits on planet-star radius ratio [len 2]
    :param depth_lims: limits on transit depth [len 2]
    :param n_trans: exact number of transits [int]
    :return: transit model matching time
    """
    iter_count = 0
    in_limits = False
    while not in_limits:    # TODO: speed-up / improve
        params = batman.TransitParams()
        params.t0 = np.random.uniform(t0_lims[0], t0_lims[1])
        params.per = np.random.uniform(per_lims[0], per_lims[1])
        params.rp = np.random.uniform(rp_lims[0], rp_lims[1])
        params.a = keplerslaw(params.per)
        params.inc = np.random.uniform(calc_i(params.a, 1.), 90.)
        params.ecc = 0.
        params.w = 90.
        params.u = [0.5079, 0.2239]
        params.limb_dark = "quadratic"

        bat = batman.TransitModel(params, time, supersample_factor=15, exp_time=29.4 / 60. / 24.0)
        model = bat.light_curve(params)

        depth = 1. - model.min()
        n_trans_model = int(np.floor((time.max() - params.t0) / params.per) + 1)

        # print depth, n_trans_model

        # in_limits = True
        in_limits = (depth_lims[0] <= depth <= depth_lims[1]) & (n_trans_model == n_trans)

        iter_count += 1

    # print iter_count, int(depth*1e6), batman_to_string(params)

    return model, params, int(depth*1e6)


def do_bls(time, flux, flux_err, per_range, q_range, nf, nbin=None, mfn=201, plot=False):
    """
    Run BLS algorithm
    :param time: LC time in days
    :param flux: LC flux normalised to 1
    :param flux_err: LC error [len time]
    :param per_range: range of periods to search [len 2]
    :param q_range: range of duration-period ratios to search [len 2]
    :param nf: number of periods to search [int]
    :param nbin: number of bins to use in phase-fold [int, None]
    :param mfn: window of median filter [int]
    :return: highest SDE and corresponding period
    """
    if not nbin:
        nbin = time.size

    bls = BLS(time, flux, flux_err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()

    p_ar, p_pow = bls.period, res.p     # power at each period
    trend = medfilt(p_pow, mfn)
    rmed_pow = p_pow - trend            # subtract running median
    p_sde = rmed_pow / rmed_pow.std()   # SDE

    sde_trans = np.nanmax(p_sde)        # highest SDE
    bper = p_ar[np.argmax(p_sde)]       # corresponding period

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(p_ar, p_sde, lw=2, zorder=3)
        plt.plot(bper, sde_trans, "*", ms=20, label="BLS period", zorder=5)
        plt.axvline(pars.per, label="Injected period", c="k", ls="-", lw=15, alpha=0.2)
        plt.xlim(0, p_ar[0])
        plt.ylim(0, int(sde_trans)+1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sde_trans, bper


n_tests = 10    # number of injections to do
tol = 0.1       # tolerance on period match in days
sde_lim = 8.    # minimum SDE to detect

n_rec, n_not = 0, 0
for i in range(n_tests):
    m, pars, depth = random_transit_model(t, (t[50], t[1500]), (20., 30.), (1e-2, 4e-2), (0.0005, 0.0015), 3)

    # plt.plot(t, m)
    # plt.show()

    f_i = f + m - 1.    # injected flux

    # plt.plot(t, f_i, ".")
    # plt.plot(t, m)
    # plt.show()

    sde, period = do_bls(t, f_i, e, (1., t[-1]-t[0]), (0.002, 0.1), 10000)

    if (sde >= sde_lim) and (pars.per-tol <= period <= pars.per+tol):
        print "> i = {:d} recovered\n".format(i)
        n_rec += 1
    else:
        print "> i = {:d} not recovered, depth = {:d} ppm".format(i, depth)
        print batman_to_string(pars)
        n_not += 1


print "{:d} ({:.2f}%) were recovered.".format(n_rec, float(n_rec)/float(n_rec+n_not)*100.)
