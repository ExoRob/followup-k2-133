# import numpy as np
# import matplotlib.pyplot as plt

# dist1 = np.random.normal(100., 10., 100)
# dist2 = np.random.normal(0.05, 0.005, 100)
#
# res1 = dist1
# res1 *= dist2
#
# res2 = dist1 * dist2
#
# print np.median(res1)
# print np.median(res2)

# 4.986893617080765
# 0.24957162692779786

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.hist(res1, bins=100)
# ax2.hist(res2, bins=100)
# plt.show()


# samples = np.array([np.random.normal(0.05, 0.005, 100),
#                     np.random.normal(10., 2., 100),
#                     np.random.normal(20., 1., 100)]).T
#
# r_sun_to_earth = 1e3 / 10.
#
# print samples.T[1] * np.random.normal(0.5, 0.1, samples.shape[0]) * r_sun_to_earth
# print samples.T[1] * np.random.normal(0.5, 0.1, samples.shape[0])

# print np.array([10.]) * np.array([5.]) * 100.


import dill
import numpy as np

with open("somesamples.pkl", "rb") as pf:
    samples = dill.load(pf)

# with open("save_1_16_150_1600_800/mcmc.pkl", "rb") as pklf:
#     t = dill.load(pklf)

print samples.T[3] * np.random.normal(.1, 0.001, samples.shape[0]) * 100.


# import dill
# # import numpy as np
#
#
# class eg():
#     def __init__(self):
#         self.a = []
#         self.b = 0
#
#
# # x = np.ndarray((2, 2), buffer=np.array([1., 2., 3., 4.]))
# # y = eg()
# # z = eg()
# # with open("save.pkl", "wb") as pf:
# #     dill.dump([z, y, x], pf)
#
# with open("save.pkl", "rb") as pf:
#     z, y, x = dill.load(pf)
# print x.T[0] * np.array([10., 10.]) * 0.1
