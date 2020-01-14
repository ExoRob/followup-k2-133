from transitleastsquares import transitleastsquares
import numpy as np
import matplotlib.pyplot as plt
import dill
from tqdm import tqdm
import batman
from my_exoplanet import phase_fold
from copy import deepcopy
import sys
sys.path.append("/Users/rwells/Documents/LDC3/python")
import LDC3


def get_transit_curve(pars, t):
    """ return the transit curve at times t """
    light_curve = np.ones(t.size, dtype=float)

    # set self.params for batman model
    params = batman.TransitParams()
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"

    c2, c3, c4 = LDC3.forward(pars[-3:])
    params.u = [0., c2, c3, c4]

    for pl in range(4):
        params.rp = pars[0+pl*4]
        params.inc = pars[1+pl*4]
        params.t0 = pars[2+pl*4]
        params.per = pars[3+pl*4]
        params.a = keplerslaw(params.per)
        batman_model = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
        light_curve += batman_model.light_curve(params) - 1.

    return light_curve


def single_model(pars, t, pl):
    # set self.params for batman model
    params = batman.TransitParams()
    params.ecc = 0.
    params.w = 90.
    params.limb_dark = "nonlinear"

    c2, c3, c4 = LDC3.forward(pars[-3:])
    params.u = [0., c2, c3, c4]

    params.rp = pars[0 + pl * 4]
    params.inc = pars[1 + pl * 4]
    params.t0 = pars[2 + pl * 4]
    params.per = pars[3 + pl * 4]
    params.a = keplerslaw(params.per)
    batman_model = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60. / 24.)
    light_curve = batman_model.light_curve(params)

    return light_curve, params


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


st_r, st_m = 0.455, 0.461
st_r_e, st_m_e = 0.022, 0.011
confs = [50.-68.27/2., 50., 50.+68.27/2.]

find_bestfit = False
run = "mcmc/save_broad_19_300_4000_2600"
# run = "mcmc/save_broad_19_250_1500_800"

pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)
rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
sas = keplerslaw(pers)
cs = np.array([planet.alpha_h, planet.alpha_r, planet.alpha_t])

inds = np.random.randint(0, samples.shape[0], 500)
if find_bestfit:
    chisq = 1e10

    for i, ind in enumerate(tqdm(inds, desc="Finding best-fit values")):
        m_full = get_transit_curve(samples[ind], data.LCtime)

        new_chisq = sum((data.LC - m_full)**2.)
        if new_chisq < chisq:
            chisq = deepcopy(new_chisq)
            bestfit_model = np.copy(m_full)
            bestfit_pars = deepcopy(samples[ind])
            # print chisq

    with open(run + "/bestfit.pkl", "wb") as pf:
        dill.dump([bestfit_pars, bestfit_model], pf)

else:
    with open(run + "/bestfit.pkl", "rb") as pf:
        bestfit_pars, bestfit_model = dill.load(pf)


t, f, e = data.LCtime, data.LC, data.LCerror
models = [single_model(bestfit_pars, data.LCtime, pl)[0] for pl in range(4)]
pars = [single_model(bestfit_pars, data.LCtime, pl)[1] for pl in range(4)]
m_tot = np.sum(models, axis=0) - 3.

for i in range(4):
    pl = ["b", "c", "d", "01"][i]
    others = [0, 1, 2, 3]
    others.remove(i)
    m_others = np.ones(m_tot.size, dtype=float)
    for j in others:
        m_others += (models[j] - 1.)

    lc = f - m_others + 1.

    # plt.plot(t, lc, ".")
    # plt.plot(t, models[i])
    # plt.show()

    model = transitleastsquares(t, lc, e)
    results = model.power(R_star=1., M_star=1., oversampling_factor=1, duration_grid_step=1.1)

    plt.figure()
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.savefig("{}-tls.pdf".format(i))
    plt.clf()
    plt.close()


# lc = f - m_tot + 1.
# model = transitleastsquares(t, lc, e)
# results = model.power(R_star=1., M_star=1., oversampling_factor=1, duration_grid_step=1.1)
#
# plt.figure()
# ax = plt.gca()
# ax.axvline(results.period, alpha=0.4, lw=3)
# plt.xlim(np.min(results.periods), np.max(results.periods))
# for n in range(2, 10):
#     ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
#     ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
# plt.ylabel(r'SDE')
# plt.xlabel('Period (days)')
# plt.plot(results.periods, results.power, color='black', lw=0.5)
# plt.xlim(0, max(results.periods))
# plt.savefig("{}-tls.pdf".format(4))
# plt.clf()
# plt.close()
#
# print(results.period)
# print(['{0:0.5f}'.format(i) for i in results.transit_times])
# print(results.depth)
# print(results.duration)
#
# plt.figure()
# plt.plot(
#     results.model_folded_phase,
#     results.model_folded_model,
#     color='red')
# plt.scatter(
#     results.folded_phase,
#     results.folded_y,
#     color='blue',
#     s=10,
#     alpha=0.5,
#     zorder=2)
# plt.show()


# t, f, e = numpy.loadtxt("final-lc-mySFF.dat", unpack=True)
# y_filt = f
# model = transitleastsquares(t, y_filt)
# results = model.power(oversampling_factor=5, duration_grid_step=1.02)
#
# plt.figure()
# ax = plt.gca()
# ax.axvline(results.period, alpha=0.4, lw=3)
# plt.xlim(numpy.min(results.periods), numpy.max(results.periods))
# for n in range(2, 10):
#     ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
#     ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
# plt.ylabel(r'SDE')
# plt.xlabel('Period (days)')
# plt.plot(results.periods, results.power, color='black', lw=0.5)
# plt.xlim(0, max(results.periods))
# plt.show()
