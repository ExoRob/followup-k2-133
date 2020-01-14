import numpy as np
import my_constants as myc
import matplotlib.pyplot as plt
nan = np.nan


def eq_teff(X, Y, a, b, c, d, e, f, g):
    return a + b * X + c * X ** 2 + d * X ** 3 + e * X ** 4 + f * Y + g * Y ** 2


def eq_star(X, Y, a, b, c, d, e, f):
    return (a + b * X + c * X ** 2 + d * X ** 3 + e * X ** 4) * (1. + f * Y)


d, derr = 75.2, .2
N = 100000
feh = -0.33

kep, B, V, u, g, r, i, z, J, H, K = \
    [13.327, 15.491, 14.063, nan, 14.804, 13.451, 12.695, nan, 11.084, 10.487, 10.279]
keperr, Berr, Verr, uerr, gerr, rerr, ierr, zerr, Jerr, Herr, Kerr = \
    [nan, 0.03, 0.036, nan, 0.038, 0.043, 0.062, nan, 0.021, 0.021, 0.018]

# print r, rerr
# print J, Jerr
# print K, Kerr

V, Verr = 14.288, 0.010
B, Berr = 16.114, 0.031

r = np.random.normal(r, rerr, N)
J = np.random.normal(J, Jerr, N)
H = np.random.normal(H, Herr, N)
K = np.random.normal(K, Kerr, N)
V = np.random.normal(V, Verr, N)
d = np.random.normal(d, derr, N)

MK = K - 5.*(np.log10(d) - 1.)

# plt.hist(MK, bins=500)
# plt.show()

# for vals in [[eq_teff(V-J, J-H, 2.769, -1.421, 0.4284, -0.06133, 0.003310, 0.1333, 0.05416) * 3500., 48.],
#              # [eq_teff(r-z, J-H, 1.384, -0.6132, 0.3110, -0.08574, 0.008895, 0.1865, -0.02039) * 3500., 55.],
#              # [eq_teff(r-J, J-H, 2.151, -1.092, 0.3767, -0.06292, 0.003950, 0.1697, 0.03106) * 3500., 52.]
#              ]:
#
#     teff, teff_sc = vals
#     pc = np.percentile(teff, [50., 5., 95.])
#     err = np.sqrt(np.average(np.abs(pc[1:] - pc[0])) + teff_sc**2. + 60.**2.)
#
#     print pc[0], err

# teff = eq_teff(V-J, J-H, 2.769, -1.421, 0.4284, -0.06133, 0.003310, 0.1333, 0.05416) * 3500.; teff_sc = 48.
# teff = eq_teff(r-z, J-H, 1.384, -0.6132, 0.3110, -0.08574, 0.008895, 0.1865, -0.02039) * 3500.; teff_sc = 55.
# teff = eq_teff(r-J, J-H, 2.151, -1.092, 0.3767, -0.06292, 0.003950, 0.1697, 0.03106) * 3500.; teff_sc = 52.
# teff = eq_teff(r-z, feh, 1.572, -0.7220, 0.3560, -0.09221, 0.009071, 0.05220, 0.) * 3500.
teff = eq_teff(r-J, feh, 2.532, -1.319, 0.4449, -0.07151, 0.004333, 0.05629, 0.) * 3500.


# rs = eq_star(MK, 0., 1.9515, -0.3520, 0.01680, 0., 0., 0.); rs_sc = 0.0289
rs = eq_star(MK, feh, 1.9305, -0.3466, 0.01647, 0., 0., 0.04458); rs_sc = 0.027
# rs = eq_star(teff/3500., 0., 10.5440, -33.7546, 35.1909, -11.59280, 0., 0.); rs_sc = 0.134
# rs = eq_star(teff/3500., feh, 16.7700, -54.3210, 57.6627, -19.69940, 0., 0.45650); rs_sc = 0.093


print np.percentile(teff, [50.-68.27/2., 50, 50.+68.27/2.]), np.percentile(rs, [50.-68.27/2., 50, 50.+68.27/2.])

# ms = eq_star(MK, 0., 0.5858, 0.3872, -0.1217, 0.0106, -2.7262e-4, 0.); ms_sc = 0.018

# print "Teff = {:.0f} K\nRs = {:.2f} Rsun\nMs = {:.2f} Msun".format(teff, rs, ms)


# (pop, add)
# for i, par in enumerate([(teff, teff_sc**2. + 60.**2.),
#                          (rs, (rs_sc*rs)**2.),
#                          (ms, (ms_sc*ms)**2.)]):
#     pc = np.percentile(par[0], [50., 5., 95.])
#
#     val = pc[0]
#     err = np.sqrt(np.average(np.abs(pc[1:] - pc[0])) + np.average(par[1]))
#     print val, err
#
#     plt.hist(par[0], bins=500)
#     plt.show()
