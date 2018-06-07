from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import pickle
from PyAstronomy.modelSuite import forTrans as ft
import sys
import dill
import my_constants as myc
from exoplanet import calc_b


st_r, st_r_e = 0.45609, 0.00345
st_m, st_m_e = 0.49699, 0.00268
# T = [3634, 3645, 3622]
# L = [0.03273 0.03362 0.03181]

R_sun, R_earth = myc.RS, myc.RE     # radii (m)
M_sun, M_earth = myc.MS, myc.ME     # masses (kg)
au = myc.AU                         # AU (m)
G = 6.67408e-11  # m3 kg-1 s-2


def keplerslaw(kper):
    """ relate parameters "a" - star-planet distance in stellar radii! and "per" - orbital period """
    Mstar = st_m * M_sun  # kg
    Rstar = st_r * R_sun  # m

    return ((kper * 86400.) ** 2 / (4 * np.pi ** 2) * G * Mstar) ** (1. / 3.) / Rstar


planet_properties = ['Period (days)', '$T_{0}$ (BJD)', 'Duration (hours)', 'Depth (ppm)', 'a (au)', '$R_{p}/R_{s}$',
                     'Radius ($R_{\earth}$)', 'Inclination (degrees)', 'Impact']

run = "save_K2SFF_16_300_2000_1000"; lc_file = "LC_K2SFF.dat"
pklfile = run + "/mcmc.pkl"
with open(pklfile, "rb") as pklf:
    data, planet, samples = dill.load(pklf)

rps = np.array([planet.rp_b, planet.rp_c, planet.rp_d, planet.rp_01])
incs = np.array([planet.inc_b, planet.inc_c, planet.inc_d, planet.inc_01])
pers = np.array([planet.period_b, planet.period_c, planet.period_d, planet.period_01])
t0s = np.array([planet.t0_b, planet.t0_c, planet.t0_d, planet.t0_01])
# sas = keplerslaw(pers)

pcs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [5, 50, 95], axis=0)))
pl_u = np.asarray(pcs).reshape(4, 4, 3)
pl_c = ["b", "c", "d", "01"]

table_full = []                                         # string for planet properties table
column_headers, pl_props, pl_prop_u, pl_prop_l = 'Property & ', {}, {}, {}
for i in range(4):
    pli = [2, 1, 0, 3][i]
    planet = pl_c[i]

    t0 = pl_u[pli][2][0]        # T0 (BJD)
    t0_l = pl_u[pli][2][2]      # 95% confidence
    t0_u = pl_u[pli][2][1]

    per = pl_u[pli][3][0]       # period (days)
    per_l = pl_u[pli][3][2]     # 95% confidence
    per_u = pl_u[pli][3][1]

    a = keplerslaw(per) * st_r * R_sun      # semi-major axis (m)
    a_au = a / au                           # semi-major axis (au)
    a_au_u = a_au * np.sqrt((1./3.*G/(4.*np.pi**2.)**(1./3.) * np.sqrt((2.*per_u/per)**2.+(st_m_e/st_m)**2.))**2. +
                            (st_r_e/st_r)**2.)
    a_au_l = a_au * np.sqrt((1./3.*G/(4.*np.pi**2.)**(1./3.) * np.sqrt((2.*per_l/per)**2.+(st_m_e/st_m)**2.))**2. +
                            (st_r_e/st_r)**2.)

    # dur = max(dmy) - min(dmy)   # transit duration (hours)
    # depth = round((max(model_ss) - min(model_ss)) * 10.0**6.0, 1)   # transit depth (ppm)
    dur, depth = 0., 0.

    rprs = pl_u[pli][0][0]      # planet-star radius ratio
    rprs_l = pl_u[pli][0][2]    # 95% confidence
    rprs_u = pl_u[pli][0][1]
    rp = rprs * st_r * R_sun / R_earth      # planetary radius (R_e)
    rp_l = rp * np.sqrt((st_r_e/st_r)**2. + (rprs_l/rprs)**2.)
    rp_u = rp * np.sqrt((st_r_e/st_r)**2. + (rprs_u/rprs)**2.)

    inc = pl_u[pli][1][0]       # inclination (degrees)
    inc_l = pl_u[pli][1][2]     # 95% confidence
    inc_u = pl_u[pli][1][1]

    inc_rad = np.radians(inc)
    inc_e_rad = np.radians(inc_u)
    a_e = 1./3.*G/(4.*np.pi**2.)**(1./3.) * np.sqrt((2.*per_u/per)**2.+(st_m_e/st_m)**2.) / keplerslaw(per)
    cosi_e = inc_e_rad*np.tan(inc_rad)

    # print a_e, cosi_e, np.tan(inc_rad)

    b = calc_b(keplerslaw(per), inc)
    b_e = b * np.sqrt(a_e**2. + cosi_e**2.)

    pl_props[planet] = ['%.4f'% per, '%.4f'% t0, '%.4f'% dur, str(depth), '%.3f'% a_au, '%.4f'% rprs, '%.2f'% rp,
                        '%.3f'% inc, '%.2f'% b]
    pl_prop_l[planet] = ['%.4f'% per_l, '%.4f'% t0_l, '', '', '%.4f'% a_au_l, '%.4f'% rprs_l,
                         '%.2f'% rp_l, '%.3f'% inc_l, '%.2f'% b_e]
    pl_prop_u[planet] = ['%.4f'% per_u, '%.4f'% t0_u, '', '', '%.4f'% a_au_u, '%.4f'% rprs_u,
                         '%.2f'% rp_u, '%.3f'% inc_u, '%.2f'% b_e]

    # rows = ['\\textbf{Planet '+pl_c[i]+'} & \\\\']
    # for j in range(len(planet_properties)):
    #     if pl_prop_l[j] != pl_prop_u[j]:
    #         err = '_{' + pl_prop_l[j] + '}^{' + pl_prop_u[j] + '}$'
    #     else:
    #         err = ' \pm ' + pl_prop_l[j] + '$'
    #     rows.append(planet_properties[j] + ' & $' + pl_props[j] + err + ' \\\\ [0.1cm]')
    # rows.append('\\\\')
    # table_full = rows + table_full


for i in range(len(pl_c)):
    column_headers += 'Planet ' + pl_c[i]
    if i != len(pl_c) - 1:
        column_headers += ' & '
    else:
        column_headers += ' \\\\'
print column_headers

for i in range(len(planet_properties)):
    row = planet_properties[i] + ' & '

    for j in range(len(pl_c)):
        planet = pl_c[j]
        val, err_u, err_l = pl_props[planet][i], pl_prop_u[planet][i], pl_prop_l[planet][i]

        if err_l != err_u:
            err = '_{-' + err_l + '}^{+' + err_u + '}$'
        else:
            err = ' \pm ' + err_l + '$'

        if j != len(pl_c) - 1:
            end = ' & '
        else:
            end = ' \\\\ [0.1cm]'

        row += '$' + val + err + end

    print row
