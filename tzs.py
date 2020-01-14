import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord, ICRS, HeliocentricTrueEcliptic, Galactocentric
import astropy.time as time
from astropy import units as u
import matplotlib.cm as cm
from tqdm import tqdm
# import os
from scipy.optimize import curve_fit
import seaborn

seaborn.set()
seaborn.set_palette("Dark2")
seaborn.set_style("ticks")
seaborn.set_color_codes()


dens = 400                  # number of data points / degree in fits
colors = cm.rainbow(np.linspace(0, 1, 8))

epoch = 2015.5      # Gaia epoch
ra = 70.1494        # deg
dec = 25.0098       # deg
pm_ra = 185.66      # mas / yr
pm_dec = -46.36     # mas / yr
rv = 96.9           # km / s
plx = 13.27         # mas
dist = 75.2         # pc


def TZ_calc(R_p, a, R_s):
    """
    Calculates the transit zone angle
    :param R_p: Planetary radius
    :param a: Sun-planet distance
    :param R_s: Solar radius
    :return: Transit zone angle
    """
    return np.degrees(2.0 * (np.arctan(R_s / a) - np.arcsin(R_p / np.sqrt(a*a + R_s*R_s))))


def fit(x, A, d):                       # fits sine curve
    x = np.radians(x)

    return A * np.sin(x - d)


def fit2(x, A, d, c):                   # fits sine curve + offset
    x = np.radians(x)

    return A * np.sin(x - d) + c


def mas2deg(mas):
    return mas / 3600. / 1e3


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Loads planetary data and compute angles"""

# data from http://solarsystem.nasa.gov/planets/
au = 149597870700.0 / 1000.0    # 1 AU (km)
R_sun = 695508.0                # Radius of Sun (km)
names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']     # names of planets
radii = [2.4397E3, 6.0518E3, 6.3710E3, 3.3895E3, 6.9911E4, 5.8232E4, 2.5362E4, 2.4622E4]    # radii of planets
s_d = [57.909227E6, 1.0820948E8, 1.4959826E8, 2.2794382E8, 7.7834082E8, 1.4266664E9,        # semi-major axis
       2.8706582E9, 4.4983964E9]

sun_distances = []          # Sun-planet distances over 1 complete orbit from JPL Horizons
for i in range(len(names)):
    a = np.genfromtxt('../K2/OrbitData/ecl_helio_'+names[i]+'.txt',
                      delimiter=',', skip_header=34, skip_footer=50)[:, 8]
    sun_distances.append(a)

psi_TZ_ar = []          # variable transit zone angle over 1 orbit
for i in range(len(names)):
    R = radii[i]                # planetary radius
    d = sun_distances[i]        # semi-major axis

    # compute angles over 1 complete orbit
    psi_TZ_ar.append([])
    for j in range(len(d)):
        psi_TZ_ar[i].append(TZ_calc(R, d[j]*au, R_sun))


# Load ecliptic data from JPL Horizons
ecl_lon_list, ecl_lat_list = [], []     # helio-centric ecliptic coordinates of the solar system planets over 1 orbit
for i in range(len(names)):
    ecl_lon_list.append(np.genfromtxt('../K2/OrbitData/ecl_helio_' + names[i] + '.txt', delimiter=',', skip_header=34,
                                      skip_footer=50)[:, 6])
    ecl_lat_list.append(np.genfromtxt('../K2/OrbitData/ecl_helio_' + names[i] + '.txt', delimiter=',', skip_header=34,
                                      skip_footer=50)[:, 7])

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Fits sinusoidal curves to the ecliptic coordinates and variable angles over 1 complete orbit
   The parameter 'dens' gives the density of the curves. I.e. 'dens' datapoints per degree"""

print '> Fitting curves to data. (dens = ' + str(dens) + ')'
data_fits = []      # holds all fits to the coordinates
fit_params = []     # holds all parameters of each fit
psi_fits = []       # transit zone angle
psi_params = []

fig = plt.figure(figsize=(15, 7))       # initialise figure
ax = fig.add_subplot(111)

for i in tqdm(range(len(names))):
    popt1, pcov1 = curve_fit(fit, ecl_lon_list[i], ecl_lat_list[i])     # fit coordinates to sine curve
    fit_params.append(popt1)

    popt2, pcov2 = curve_fit(fit2, ecl_lon_list[i], psi_TZ_ar[i])       # transit zone angle
    psi_params.append(popt2)

    data_fit = []
    psi_fit = []
    x_fit = []      # longitude for fit

    for j in range(360 * dens):
        data_fit.append(fit(j / float(dens), popt1[0], popt1[1]))
        psi_fit.append(fit2(j / float(dens), popt2[0], popt2[1], popt2[2]))
        x_fit.append(j / float(dens))

    psi_fits.append(psi_fit)
    data_fits.append(data_fit)

    if i != 2:          # colours on plot - Earth as black
        c = colors[i]
    else:
        c = 'black'
    df1 = data_fit + np.asarray(psi_fits[i]) / 2.0      # upper transit zone boundary
    df2 = data_fit - np.asarray(psi_fits[i]) / 2.0      # lower transit zone boundary

    # sample boundaries for smaller filesize of plot
    x_fit_c, df1_c, df2_c = [], [], []
    for k in range(0, len(x_fit), dens/25):
        x_fit_c.append(x_fit[k])
        df1_c.append(df1[k])
        df2_c.append(df2[k])
    x_fit_c, df1_c, df2_c = np.asarray(x_fit_c), np.asarray(df1_c), np.asarray(df2_c)

    ax.fill_between(x_fit_c, df1_c, df2_c, where=df1_c >= df2_c, edgecolor=c, facecolor=c, alpha=0.4,
                    interpolate=True, label=names[i])

ax.set_xlabel('Longitude (Degrees)', fontsize=15)
ax.set_ylabel('Latitude (Degrees)', fontsize=15)
ax.set_xlim(0., 360.)
ax.legend(loc='upper left', fontsize=15)
plt.tight_layout()

#

# TODO: compute real velocity -> new coords?
# TODO: plot as arrow

icrs = ICRS(ra=ra * u.degree, dec=dec * u.degree, distance=dist * u.pc,
            pm_ra_cosdec=pm_ra * u.mas / u.yr, pm_dec=pm_dec * u.mas / u.yr, radial_velocity=rv * u.m * 1e3 / u.s)
gal = icrs.transform_to(Galactocentric)

print icrs.transform_to(HeliocentricTrueEcliptic)
print gal.transform_to(HeliocentricTrueEcliptic)


ecl_lons, ecl_lats = [], []
for n_yrs in tqdm(np.arange(0, 100000, 10000)):
    ecl_n = SkyCoord(ra+n_yrs*mas2deg(pm_ra), dec+n_yrs*mas2deg(pm_dec), unit=(u.degree, u.degree), frame='icrs',
                     equinox='J2000.0', distance=dist * u.pc).heliocentrictrueecliptic

    # print "{}-01-01T00:00:00".format(2000 + n_yrs)
    # ecl_n = icrs.transform_to(HeliocentricTrueEcliptic(
    #     obstime=time.Time("{}-01-01T00:00:00".format(2000 + n_yrs), format='isot', scale='utc')))

    # ecl_n = Galactocentric(x=gal.x + n_yrs*u.yr * gal.v_x,
    #                        y=gal.y + n_yrs*u.yr * gal.v_y,
    #                        z=gal.z + n_yrs*u.yr * gal.v_z).transform_to(HeliocentricTrueEcliptic)

    ecl_lons.append(ecl_n.lon.deg)
    ecl_lats.append(ecl_n.lat.deg)

plt.plot(ecl_lons, ecl_lats, color="k", lw=3)
plt.show()
