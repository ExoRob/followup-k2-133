import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as tic
# from astropy.io import fits
# import batman
# from tqdm import tqdm
# import os
# from scipy.interpolate import interp1d
# import my_constants as myc
from scipy.ndimage import median_filter
import transits
from pybls import BLS
import seaborn as sns
sns.set()
sns.set_palette("Dark2")
sns.set_color_codes()
sns.set_style("ticks")

t, f, e = np.loadtxt("lc-none-mySFF.dat", unpack=True)

sample = np.random.randint(0, t.size, t.size)

n = np.array([np.sum(sample == i) for i in range(t.size)])

new_e = e / np.sqrt(n.astype(float))
mask = np.isfinite(new_e)
new_t, new_f, new_e = t[mask], f[mask], new_e[mask]


bls = BLS(t, f, e, period_range=(1., 40.), q_range=(0.001, 0.1), nf=50000, nbin=t.size)
res = bls()
p_ar, p_pow = bls.period, res.p  # power at each period
rmed_pow = p_pow - median_filter(p_pow, 1501)
p_sde = rmed_pow / rmed_pow.std()  # SDE
sde_0 = np.nanmax(p_sde)
per_0 = p_ar[np.argmax(p_sde)]

print sde_0, per_0
plt.plot(p_ar, p_sde)

bls = BLS(new_t, new_f, new_e, period_range=(1., 40.), q_range=(0.001, 0.1), nf=50000, nbin=t.size)
res = bls()
p_ar, p_pow = bls.period, res.p  # power at each period
rmed_pow = p_pow - median_filter(p_pow, 1501)
p_sde = rmed_pow / rmed_pow.std()  # SDE
sde_rs = np.nanmax(p_sde)
per_rs = p_ar[np.argmax(p_sde)]

print sde_rs, per_rs
plt.plot(p_ar, p_sde)
plt.show()

# plt.errorbar(t, f, new_e, lw=0., marker=".", ms=8, elinewidth=1.5, color="k", zorder=2)
# plt.show()
