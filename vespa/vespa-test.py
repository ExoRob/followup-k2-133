import vespa
from isochrones import StarModel
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import dill

# target
epic = 248690431
ra = 156.113404
dec = 8.113607
rprs = 0.136313770331
period = 0.253769272327

Kepler = 14.955, None
B = 17.022, 0.07
V = 15.589, 0.055
u = None, None
g = 16.43, 0.01
r = 15.054, 0.05
i = 14.323, 0.05
z = None, None
J = 12.803, 0.028
H = 12.137, 0.024
K = 11.957, 0.026
W1 = 11.826, 0.023
W2 = 11.871, 0.022
W3 = 11.451, 0.323
W4 = 8.318, None

if not os.path.exists("transit.pkl"):
    t, f, e = np.loadtxt("{}/phase_lc.dat".format(epic), unpack=True, delimiter=",")

    transit = vespa.TransitSignal(ts=t, fs=f, dfs=e, P=period)
    transit.MCMC(niter=1000, nburn=500, nwalkers=200)
    transit.save_pkl("transit.pkl")

    transit.plot(plot_trap=True)
    plt.show()

    transit.triangle()
    plt.show()

else:
    with open("transit.pkl", "rb") as pkl:
        transit = dill.load(pkl)

if not os.path.exists("starfield.h5"):
    vespa.stars.trilegal.get_trilegal("starfield.h5", ra=ra, dec=dec, binaries=False)

mags = {}
for band, val in zip("Kepler,B,V,u,g,r,i,z,J,H,K,W1,W2,W3".split(","),
                     [Kepler, B, V, u, g, r, i, z, J, H, K, W1, W2, W3]):
    if val[0]:
        mags[band] = val

popset = vespa.PopulationSet(starmodel=StarModel.load_hdf("test-1.h5"),
                             binary_starmodel=StarModel.load_hdf("test-2.h5"),
                             triple_starmodel=StarModel.load_hdf("test-3.h5"),
                             period=period, rprs=rprs, n=2e4, ra=ra, dec=dec, trilegal_filename="starfield.h5",
                             mags=mags)   # savefile="popset.pkl",

ffpcalc = vespa.FPPCalculation(transit, popset)

ffpcalc.FPP()

ffpcalc.FPPplots(format="pdf")

ffpcalc.FPPsummary(saveplot=True, figformat="pdf")
