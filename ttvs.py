import my_constants as myc
import numpy as np
import matplotlib.pyplot as plt
from ttvfaster import run_ttvfaster
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")


def mrrelation(rp):
    if rp > 1.5:
        return 2.69 * rp**0.93
    else:
        return rp**3.


def calc_K(Mp, Ms, a, i=90.0, e=0.0):
    """
    Computes RV semi-amplitude along line of sight
    :param Mp: planetary mass (kg)
    :param Ms: stellar mass (kg)
    :param a: semi-major axis (m)
    :param i: inclination to line of sight (~90)
    :param e: eccentricity
    :return: semi-amplitude
    """
    G = 6.674e-11
    i = np.radians(i)

    return (G / (1.0 - e**2.0))**0.5 * Mp * np.sin(i) * (Mp + Ms)**(-0.5) * a**(-0.5)


def keplerslaw(kper):
    Mstar = 0.461 * 1.989e30  # kg
    # Rstar = 0.455 * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * myc.G * Mstar) ** (1. / 3.)


# params = [
#     1.0,  # M_star
#     # m1 p1 e1*cos(arg peri1) i1 Omega1 e1*sin(arg peri1) TT1
#     0.00001027, 66.03300476, -0.00654273, 1.57079637, -1.57079637, -0.05891280, 142.63816833,
#     # m2 p2 e2*cos(arg peri2) i2 Omega2 e2*sin(arg peri2) TT2
#     0.00041269, 125.85229492, -0.00096537, 1.57079637, -1.57079637, -0.00953292, 236.66268921
# ]
#
# run = run_ttvfaster(2, params, 0.0, 1600.0, 6)
#
# ttvs1 = np.asarray(run[0]) - np.arange(len(run[0]))*66.03300476
# ttvs2 = np.asarray(run[1]) - np.arange(len(run[1]))*125.85229492
#
# plt.plot(run[0], ttvs1 - np.mean(ttvs1))
# plt.plot(run[1], ttvs2 - np.mean(ttvs2))
# plt.show()

pl_chars = ["b", "c", "d", "e"]
# pl_masses = np.array([5., 5., 5., 5.]) * myc.ME / myc.MS    # M_Sun
pl_radii = np.array([1.30, 1.56, 1.94, 1.82])
pl_periods = np.array([3.0714, 4.8678, 11.0243, 26.5848])     # days
pl_epochs = np.array([2988.3150, 2990.7681, 2993.1720, 3004.8649])
pl_eccs = [0., 0., 0., 0.]                          # circular
pl_incs = np.array([87.74, 88.33, 89.73, 89.183])             # degrees
# pl_lns = [90., 90., 90., 90.]                       # ??
# pl_args = [90., 90., 90., 90.]                      # ??
# pl_mas = [90., 90., 90., 90.]                       # ??

pl_masses = np.array([mrrelation(rp) for rp in pl_radii])
print pl_masses
print calc_K(pl_masses*myc.ME, 0.461*myc.MS, keplerslaw(pl_periods), pl_incs)
pl_masses *= myc.ME / myc.MS

n_pls = len(pl_masses)

# m1 p1 e1*cos(arg peri1) i1 Omega1 e1*sin(arg peri1) TT1
params = [0.461]
for i in range(n_pls):
    params += [pl_masses[i], pl_periods[i], 0., np.radians(pl_incs[i]), 0., 0., pl_epochs[i]]

run = run_ttvfaster(4, params, 0., 10000., 10)

for i, pl_times in enumerate(run):
    ttvs = np.asarray(pl_times) - np.arange(len(pl_times)) * pl_periods[i]
    ttvs -= np.mean(ttvs)
    ttvs *= 24.*3600.

    ampl = np.max(np.abs(ttvs))
    print "{}: {:.1f} s".format(pl_chars[i], ampl)

    plt.plot(pl_times, ttvs, label=pl_chars[i])
plt.legend()
plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --
# ttvfast method
# import ttvfast
# planets = [ttvfast.models.Planet(pl_masses[i], pl_periods[i], pl_eccs[i], pl_incs[i],
#                                  pl_lns[i], pl_args[i], pl_mas[i])
#            for i in range(n_pls)]
#
# # list of planet classes
# # the stellar mass in units of solar mass,
# # the start point of the integration in days,
# # the time step for the integration in days,
# # and the end point for integration in days.
# run = ttvfast.ttvfast(planets, 0.461, 0., 0.01, 100.)
#
# indices, epochs, times, rsky, vsky = run["positions"]
#
# plt.plot(vsky)
# plt.show()
