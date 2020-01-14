import numpy as np
import matplotlib.pyplot as plt

# t0, per = 3004.8657, 26.5841
#
# t, f, e = np.loadtxt("/Users/rwells/Desktop/followup-k2-133/reprocessed/final-lc-mySFF-cut-transits.dat", unpack=True)
#
# # plt.plot(t, f, ".")
# # plt.show()
#
# p = ((t - t0 + per/2.) / per) % 1
#
# # plt.plot(p, f, ".")
# # plt.show()
#
# p -= 0.5
# p *= per
#
# np.savetxt("transit.txt", np.array([p, f, e]).T)
#
# plt.plot(p, f, ".")
# plt.show()


import pandas

df = pandas.read_csv("results_bootstrap.txt", delimiter=" ")

s = 0.
for col in df.columns:
    if col[:2] == "pr":
        s += np.mean(df[col])*100
        print col, np.mean(df[col])*100     # , np.std(df[col])

print np.round(np.array([0.07328 + 0.005964, 2.133 + 0.14222, 0.0001105 + 0.0, 97.65]) / s * 100., 2)

print s
