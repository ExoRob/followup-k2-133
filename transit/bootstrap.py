import numpy as np
from pybls import BLS
import dill
from scipy.signal import medfilt
import multiprocessing
from astropy.stats import bootstrap
from tqdm import tqdm

t, f = np.loadtxt("01_lc.dat", delimiter=",", unpack=True)
err = np.ones(t.size, dtype=float)*np.std(f)

nb = 40          # number of bootstrap resamples
ncores = 4      # number of concurrent processes (number of available cores)
first_run = False

per_range = (15., 45.)  # BLS period range
q_range = (0.002, 0.1)  # BLS q range
nf = 10000              # BLS frequency bins
nbin = t.size           # BLS phase bins


if first_run:   # compute SDE of candidate with BLS parameters set
    bls = BLS(t, f, err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()
    p_ar, p_pow = bls.period, res.p  # power at each period
    trend = medfilt(p_pow, 201)
    rmed_pow = p_pow - trend  # subtract running median
    p_sde = rmed_pow / rmed_pow.std()  # SDE
    cand_sde = np.nanmax(p_sde)  # highest SDE
    cand_per = p_ar[np.argmax(p_sde)]  # corresponding period
    print "> SDE of candidate = {}".format(cand_sde)
    import sys; sys.exit()

cand_sde = 10.0243738544

f_shuff_all = bootstrap(f, nb)  # resampled fluxes [nb * len(t)]
pbar = tqdm(total=nb, initial=0, desc="Running bootstrap")


def do_bls(f_bls):
    # do BLS search
    bls = BLS(t, f_bls, err, period_range=per_range, q_range=q_range, nf=nf, nbin=nbin)
    res = bls()

    p_ar, p_pow = bls.period, res.p     # power at each period
    trend = medfilt(p_pow, 201)
    rmed_pow = p_pow - trend            # subtract running median
    p_sde = rmed_pow / rmed_pow.std()   # SDE

    sde_trans = np.nanmax(p_sde)        # highest SDE
    bper = p_ar[np.argmax(p_sde)]       # corresponding period

    pbar.update(ncores)     # update progress once complete

    return sde_trans, bper


pool = multiprocessing.Pool(processes=ncores)
sde_ar, per_ar = np.asarray(pool.map(do_bls, f_shuff_all)).T    # run BLS for each bootstrap resample
pbar.close()

# print sde_ar
# print per_ar

with open("bootstrap_res.pkl", "wb") as pf:     # save results of bootstrap
    dill.dump([cand_sde, sde_ar, per_ar], pf)

nhigher = np.sum(sde_ar >= cand_sde)    # compute FPP from number with SDE above candidate's
print "{}/{} > candidate SDE, FPP = {}%".format(nhigher, nb, float(nhigher)/nb*100.)
