import my_constants as myc
import numpy as np
from astropy.io import ascii
from scipy import interpolate
import matplotlib.pyplot as plt

epic = 247887989
gaia_source_id = 148080473682357376

ra, dec = 70.148567, 25.010006

plx = 13.27430943170923
plx_err = 0.0353490593079183

# d, d_l, d_h = 75.1702629624071, 74.9695484897805, 75.3720358649995
# d = np.array([75.1702629624071, 74.9695484897805, 75.3720358649995])
d = np.array([75.1702629624071, 75.3720358649995, 74.9695484897805])

# md = 1.931e-20
# print np.sqrt(md * (d*myc.PC)**2.) / myc.RS

J_app = np.array([11.084, 11.084-0.021, 11.084+0.021])
K_app = np.array([10.279, 10.279-0.018, 10.279+0.018])

J_abs = J_app - 5. * np.log10(d / 10.)
K_abs = K_app - 5. * np.log10(d / 10.)

mam = ascii.read('mamajek_asciiread2.txt', fill_values=[('....', '0.0'), ('...', '0.0')])
Tgrid = mam['Teff']
Mgrid = mam['Msun']
logLgrid = mam['logL']
Jgrid = mam['M_J']
Kgrid = mam['M_Ks']
Rgrid = mam['R_Rsun']

mask = (Tgrid > 3000.) & (Tgrid < 4000.)
Kmags, Jmags, Rs, Ts, Ms, logLs = Kgrid[mask], Jgrid[mask], Rgrid[mask], Tgrid[mask], Mgrid[mask], logLgrid[mask]

JR = interpolate.interp1d(Jmags, Rs)
KR = interpolate.interp1d(Kmags, Rs)
JT = interpolate.interp1d(Jmags, Ts)
KT = interpolate.interp1d(Kmags, Ts)
JM = interpolate.interp1d(Jmags, Ms)
KM = interpolate.interp1d(Kmags, Ms)
JL = interpolate.interp1d(Jmags, logLs)
KL = interpolate.interp1d(Kmags, logLs)

r_jmag = JR(J_abs)
r_kmag = KR(K_abs)
t_jmag = JT(J_abs)
t_kmag = KT(K_abs)
m_jmag = JM(J_abs)
m_kmag = KM(K_abs)
logL_jmag = JL(J_abs)
logL_kmag = KL(K_abs)

# print np.round(r_jmag, 3)
# print np.round(r_kmag, 3)
print "R =", np.round(r_jmag/2.+r_kmag/2., 5)

# print [np.int(t) for t in t_jmag]
# print [np.int(t) for t in t_kmag]
print "T =", [np.int(t/2) for t in t_kmag+t_jmag]

# print np.round(m_jmag, 3)
# print np.round(m_kmag, 3)
print "M =", np.round(m_jmag/2.+m_kmag/2., 5)

# print np.round(10.**logL_jmag, 4)
# print np.round(10.**logL_kmag, 4)
print "L =", np.round(10.**(logL_kmag/2.+logL_jmag/2.), 5)


# jplot = np.linspace(Jmags[0], Jmags[-1], 1e4)
# plt.plot(Jmags, Rs)
# plt.plot(jplot, JR(jplot))
# plt.axvline(J_abs[0])
# plt.axhline(r_jmag[0])
# plt.axvspan(J_abs[1], J_abs[2], alpha=0.5)
# plt.axhspan(r_jmag[1], r_jmag[2], alpha=0.5)
# plt.show()

# JL = interpolate.interp1d(Jgrid, logLgrid)
# sigma = 5.6704e-5 # erg /s / cm62 / K64
# Lbolsun = 3.8270e33 # erg/s
# Lbol = 10**(logL) * Lbolsun
# Rsun = 6.957e10 # cm
# R = np.sqrt(Lbol/(4*np.pi*sigma*np.array([Teff, Teff+Tefferr, Teff-Tefferr])**4))/Rsun
# dist = np.sqrt(2.5**(J-MJ)) * 10.


# https://arxiv.org/abs/1804.10121v1
# http://gaia.ari.uni-heidelberg.de/tap.html
