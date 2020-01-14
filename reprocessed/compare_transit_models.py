import batman
from starry.kepler import Primary, Secondary, System
# import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt


def keplerslaw(kper, st_r, st_m):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * 1.989e30  # kg
    G = 6.67408e-11  # m3 kg-1 s-2
    Rstar = st_r * 695700000.  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


def calc_b(_a, _i):
    return _a * np.cos(np.radians(_i))


def calc_i(_a, _b):
    return np.degrees(np.arccos(_b / _a))


r_star = 0.45
m_star = 0.5
u_star = [0.5079, 0.2239]

period = 25.5
t0 = 3004.5
a = keplerslaw(period, r_star, m_star)
b = 0.9
r_pl = 0.035

t = np.linspace(t0-0.2, t0+0.2, 1000)

# orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b)  # ecc=ecc, omega=omega)
# light_curves = xo.StarryLightCurve(np.asarray(u_star)).\
#                    get_light_curve(orbit=orbit, r=r_pl, t=t, texp=29.4/60./24., oversample=15, use_in_transit=False)
# print light_curves.__dict__


star = Primary()
star[1], star[2] = u_star
planet = Secondary()
planet.lambda0 = 90.
planet.tref = t0
planet.r = r_pl
planet.a = a
planet.inc = calc_i(a, b)
planet.porb = period
system = System(star, planet)
system.exposure_time = 29.4/60./24.
system.exposure_max_depth = 15
system.compute(t)

params = batman.TransitParams()
params.t0 = t0
params.per = period
params.rp = r_pl
params.a = a
params.inc = calc_i(a, b)
params.ecc = 0.
params.w = 90.
params.u = u_star
params.limb_dark = "quadratic"
bat = batman.TransitModel(params, t, supersample_factor=15, exp_time=29.4/60./24.0)
m = bat.light_curve(params)

# plt.plot(t, system.lightcurve, label="Starry")
# plt.plot(t, m, label="Batman")
plt.plot(t, 1e6*(system.lightcurve - m), label="Diff")
plt.legend()
plt.show()
