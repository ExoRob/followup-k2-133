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

t, f, err = np.loadtxt("lc_none-SFF.dat", unpack=True, dtype=float)

nb = 2000       # number of bootstrap resamples
ncores = 20     # number of concurrent processes (number of available cores)
first_run = False

per_range = (1., (t[-1] - t[0]) / 2.)  	    # BLS period range
q_range = (1./24./per_range[1], 5./24.)     # BLS q range
nf = 50000                                 # BLS frequency bins
nbin = t.size                               # BLS phase bins
binsize = 1501                              # median filter kernel size

if first_run:   # compute SDE of candidate with BLS parameters set
    t, f, err = np.loadtxt("lc_01-SFF.dat", unpack=True, dtype=float)

    bls = BLS(t, f, err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()
    p_ar, p_pow = bls.period, res.p  # power at each period

    rmed_pow = p_pow - median_filter(p_pow, binsize)
    p_sde = rmed_pow / rmed_pow.std()  # SDE

    cand_sde = np.nanmax(p_sde)  # highest SDE
    cand_per = p_ar[np.argmax(p_sde)]  # corresponding period

    print "> SDE of candidate = {}".format(cand_sde)
    # plt.plot(p_ar, p_sde)
    # plt.show()
    import sys; sys.exit()

cand_sde = 10.7258695986

n_all = np.array([np.sum(np.random.randint(0, t.size, t.size*nb) == i) for i in range(t.size*nb)]).reshape(t.size, nb).T
pbar = tqdm(total=nb, initial=0, desc="Running bootstrap")


def do_bls(n):
    # do BLS search
    new_e = err / np.sqrt(n.astype(float))      # new errors from bootstrap
    mask = np.isfinite(new_e)                   # remove infinite error points
    new_t, new_f, new_e = t[mask], f[mask], new_e[mask]

    bls = BLS(new_t, new_f, new_e, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()

    p_ar, p_pow = bls.period, res.p     # power at each period
    rmed_pow = p_pow - median_filter(p_pow, binsize)
    p_sde = rmed_pow / rmed_pow.std()   # SDE

    sde_trans = np.nanmax(p_sde)        # highest SDE
    bper = p_ar[np.argmax(p_sde)]       # corresponding period

    # plt.plot(p_ar, p_sde)
    # plt.show()

    pbar.update(ncores)     # update progress once complete

    return sde_trans, bper


pool = multiprocessing.Pool(processes=ncores)
sde_ar, per_ar = np.asarray(pool.map(do_bls, n_all)).T    # run BLS for each bootstrap resample
pbar.close()

# sde_ar, per_ar = np.asarray([do_bls(fl) for fl in f_shuff_all])    # run BLS for each bootstrap resample
# print sde_ar
# print per_ar

i = 0
while os.path.exists("bootstrap_res{}.pkl".format(i)):
    i += 1

with open("bootstrap_res{}.pkl".format(i), "wb") as pf:     # save results of bootstrap
    dill.dump([cand_sde, sde_ar, per_ar], pf)

nhigher = np.sum(sde_ar >= cand_sde)    # compute FPP from number with SDE above candidate's
print "{}/{} > candidate SDE, FPP = {}%".format(nhigher, nb, float(nhigher)/nb*100.)

