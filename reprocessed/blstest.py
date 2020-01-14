import numpy as np
from pybls import BLS
import dill
import os
# from scipy.signal import medfilt
from scipy.ndimage import median_filter
import multiprocessing
from astropy.stats import bootstrap
from tqdm import tqdm
import matplotlib.pyplot as plt


def do_bls(new_t, new_f, new_e, per_range, q_range, nf, nbin, binsize):
    # do BLS search
    bls = BLS(new_t, new_f, new_e, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()

    p_ar, p_pow = bls.period, res.p     # power at each period
    rmed_pow = p_pow - median_filter(p_pow, binsize)
    p_sde = rmed_pow / rmed_pow.std()   # SDE

    sde_trans = np.nanmax(p_sde)        # highest SDE
    bper = p_ar[np.argmax(p_sde)]       # corresponding period

    # plt.plot(p_ar, p_sde)
    # plt.show()

    return sde_trans, bper


for lcname in ["lc_01-SFF.dat", "lc_none-SFF.dat"]:
    t, f, err = np.loadtxt(lcname, unpack=True, dtype=float)

    per_range = (1., (t[-1] - t[0]) / 2.)  # BLS period range
    # per_range = (1., t[-1] - t[0] + 1.)  # BLS period range
    q_range = (1. / 24. / per_range[1], 5. / 24.)  # BLS q range
    nf = 50000  # BLS frequency bins
    nbin = t.size  # BLS phase bins
    binsize = 1501  # median filter kernel size

    sde, per = do_bls(t, f, err, per_range, q_range, nf, nbin, binsize)

    print "> SDE = {}".format(sde)
