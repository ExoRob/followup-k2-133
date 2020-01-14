import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import corner
import my_constants as myc
import seaborn
seaborn.set()
seaborn.set_style("ticks")
seaborn.set_palette("Dark2")
seaborn.set_color_codes()


def calc_s_eff(t_star, s_eff_sol, a, b, c, d):
    """
    Eq(4) - http://iopscience.iop.org/article/10.1088/2041-8205/787/2/L29/meta
    :params: Table 1 coefficients for a given T_eff, zone and planetary mass
    :return: stellar flux at HZ edge
    """
    return s_eff_sol + a*t_star + b*t_star**2. + c*t_star**3. + d*t_star**4.


def hz_d(l_star, t_star, paper="kop", zones=None, mass=1):
    """
    :param l_star: stellar luminosity in L_Sun
    :param t_star: stellar effective temperature in K
    :param zones: HZ limits (recent Venus, runaway greenhouse, maximum greenhouse, early Mars) = rv, rg, mg, em
    :param mass: planetary mass in M_Earth = 0.1, 1, 5
    :return: HZ distance from star in AU
    """
    t_use = t_star - 5780.

    if zones is None:
        zones = ["rv", "rg", "mg", "em"]

    # zone: s_eff_sol, a, b, c, d
    if paper == "kop":
        if mass == 1:    # http://iopscience.iop.org/article/10.1088/2041-8205/787/2/L29/meta
            pars = {"rv": [1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15],
                    "rg": [1.107, 1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15],
                    "mg": [0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16],
                    "em": [0.32, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16]}
        if mass == 5:
            pars = {"rg": [1.188, 1.433e-4, 1.707e-8, -8.968e-12, -2.084e-15]}

    elif paper == "ram":    # http://iopscience.iop.org/article/10.3847/1538-4357/aab8fa/meta
        pars = {"rv": [1.768, 1.3151e-4, 5.8695e-10, -2.8895e-12, 3.2174e-16],
                "rg": [1.1105, 1.1921e-4, 9.5932e-9, -2.6189e-12, 1.3710e-16],
                "mg": [0.3587, 5.8087e-5, 1.5393e-9, -8.3547e-13, 1.0319e-16],
                "em": [0.3246, 5.213e-5, 4.5245e-10, -1.0223e-12, 9.6376e-17]}

    s_eff = np.array([calc_s_eff(t_use, *pars[zone]) for zone in zones])

    return np.sqrt(l_star / s_eff)


def hz_d_phl(l_star, t_star, zone=None):
    # http://phl.upr.edu/library/notes/habitablezonesdistancehzdahabitabilitymetricforexoplanets
    ai, bi, ao, bo = 2.7619e-5, 3.8095e-9, 1.3786e-4, 1.4286e-9
    t_use = t_star - 5700.

    if zone is None:
        zone = "mv"

    pars = {"mv": [0.72, 1.77],
            "c0": [0.84, 1.67],
            "c50": [0.68, 1.95],
            "c100": [0.46, 2.40]}
    ris, ros = pars[zone]

    ri = (ris - ai * t_use - bi * t_use**2.) * np.sqrt(l_star)
    ro = (ros - ao * t_use - bo * t_use**2.) * np.sqrt(l_star)

    return ri, ro


def calc_s(teff, rs, d):
    s_star = 5.67e-8 * teff**4.   # W / m2
    s_pl = s_star * (rs / d)**2.

    return s_pl


def calc_esi(rp, s):
    esi = 1. - np.sqrt(0.5 * (((s - 1.) / (s + 1.))**2. + ((rp - 1.) / (rp + 1.))**2.))

    return esi


def calc_teq(Ts, Rs, a, A=0.3):

    return Ts * (1. - A)**0.25 * (Rs / 2. / a)**0.5


confs = [50., 50.-68.27/2., 50.+68.27/2.]

# k2-133
st_r, st_r_e = 0.455, 0.022         # stellar radius
st_m, st_m_e = 0.461, 0.011         # stellar mass
st_l, st_l_e = 0.0332, 0.0013       # stellar luminosity
st_t, st_t_e = 3655., 80.           # stellar effective temperature
pl_r, pl_r_e = 1.731, 0.14          # planetary radius
a_au, a_au_e = 0.13463, 0.00107     # orbital semi-major axis


# k2-288 B
# st_r, st_r_e = 0.32, 0.03           # stellar radius
# st_m, st_m_e = 0.33, 0.02           # stellar mass
# st_l, st_l_e = 0.01175, 0.00055     # stellar luminosity
# st_t, st_t_e = 3341., 276.          # stellar effective temperature
# pl_r, pl_r_e = 1.9, 0.3             # planetary radius
# a_au, a_au_e = 0.164, 0.03          # orbital semi-major axis


n_samples = 100000
l_sample = np.random.normal(st_l, st_l_e, n_samples)
t_sample = np.random.normal(st_t, st_t_e, n_samples)
r_sample = np.random.normal(st_r, st_r_e, n_samples)
a_sample = np.random.normal(a_au, a_au_e, n_samples)
rp_sample = np.random.normal(pl_r, pl_r_e, n_samples)

print np.median(hz_d(l_sample, t_sample, paper="kop", mass=1), axis=1) * myc.AU / (0.455 * myc.RS)
exit()

zones = ["rv"]
zone_boundaries = hz_d(l_sample, t_sample, zones=zones, paper="kop", mass=1)
# zones = ["rv", "em"]
# zone_boundaries = hz_d_phl(l_sample, t_sample, "mv")

diff = zone_boundaries[0] - a_sample    # diff < 0 is within HZ
print "Probability in HZ = {:.1f}%".format(float(sum(diff <= 0.)) / diff.size * 100.)

# dists = np.array([l_sample, t_sample, a_sample] + [zone_boundaries[i] for i in range(len(zone_boundaries))] + [diff])
# corner.corner(dists.T, labels=["L", "T", "a"]+zones+["diff"], truths=[None]*len(dists[:-1])+[0.], bins=50)
# plt.show()

# http://iopscience.iop.org/article/10.3847/1538-4357/aa7cf9/meta#apjaa7cf9s4


s_earth = calc_s(5800., myc.RS, myc.AU)
s = calc_s(t_sample, r_sample*myc.RS, a_sample*myc.AU) / s_earth
esi = calc_esi(rp_sample, s)
tp_sample = calc_teq(t_sample, r_sample*myc.RS, a_sample*myc.AU)

print s.size, sum(np.isfinite(s))

plt.hist(s, bins=np.linspace(0., 2.5, 200))
plt.show()

s_pc = np.percentile(s, confs)
esi_pc = np.percentile(esi, confs)
tp_pc = np.percentile(tp_sample, confs)
print "S    =", s_pc[0], np.abs(s_pc[0]-s_pc[1:])
print "ESI  =", esi_pc[0], np.abs(esi_pc[0]-esi_pc[1:])
print "T_eq =", tp_pc[0], np.abs(tp_pc[0]-tp_pc[1:])

dists = np.array([r_sample, t_sample, a_sample, rp_sample, s, esi, tp_sample])
corner.corner(dists.T, labels=["Rs", "Teff", "a", "Rp", "S", "ESI", "Tp"], bins=50)
plt.show()
