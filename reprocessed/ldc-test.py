import batman
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/rwells/Documents/LDC3/python")
import LDC3

# fit alphas-h,r,t u[0,1]
# convert to LDCs - c2,c3,c4
# test if physical
# use sing LD law - non-linear with c1=0

alphas = [0.4475975539817997, 0.7294067670007804, 0.6557480380299435]   # sing 3-param kepler 4000K
ldcs = LDC3.forward(alphas)

params = batman.TransitParams()
params.t0 = 0.
params.per = 1.
params.rp = 0.1
params.a = 15.
params.inc = 87.
params.ecc = 0.
params.w = 90.
params.limb_dark = "nonlinear"
params.u = [0.] + ldcs

t = np.linspace(-0.025, 0.025, 1000)

for j in range(100):
    alphas = [np.random.uniform(alphas[i]-0.05, alphas[i]+0.05) for i in range(3)]  # random alphas
    ldcs = LDC3.forward(alphas)  # convert to c2, c3, c4
    passed = LDC3.criteriatest(0, ldcs)  # test against criteria

    if passed:
        params.u = [0.] + ldcs
        m = batman.TransitModel(params, t)
        flux = m.light_curve(params)

        plt.plot(t, flux, c="0.4", alpha=0.5)
plt.show()

# c2s, c3s, c4s = [], [], []
# for j in range(1000):
#     alphas = [stats.uniform(0.0, 1.0).rvs() for i in range(3)]  # random alphas
#     ldcs = LDC3.forward(alphas)  # convert to c2, c3, c4
#     passed = LDC3.criteriatest(0, ldcs)  # test against criteria
#
#     if passed:
#         c2s.append(ldcs[0])
#         c3s.append(ldcs[1])
#         c4s.append(ldcs[2])
#
# plt.hist(c2s, bins=20)
# plt.show()
# plt.hist(c3s, bins=20)
# plt.show()
# plt.hist(c4s, bins=20)
# plt.show()
#
# passed = False
# while not passed:
#     alphas = [np.random.uniform(0.0, 1.0) for i in range(3)]    # random alphas
#     ldcs = LDC3.forward(alphas)                                 # convert to c2, c3, c4
#     passed = LDC3.criteriatest(0, ldcs)                         # test against criteria
#
# params = batman.TransitParams()
# params.t0 = 0.
# params.per = 1.
# params.rp = 0.1
# params.a = 15.
# params.inc = 87.
# params.ecc = 0.
# params.w = 90.
# params.limb_dark = "nonlinear"
# # params.u = [0.] + ldcs
#
# t = np.linspace(-0.025, 0.025, 1000)
#
#
# for j in range(100):
#     alphas = [np.random.uniform(0.0, 1.0) for i in range(3)]  # random alphas
#     ldcs = LDC3.forward(alphas)  # convert to c2, c3, c4
#     passed = LDC3.criteriatest(0, ldcs)  # test against criteria
#
#     if passed:
#         params.u = [0.] + ldcs
#         m = batman.TransitModel(params, t)
#         flux = m.light_curve(params)
#
#         plt.plot(t, flux, c="0.4", alpha=0.5)
# plt.show()

