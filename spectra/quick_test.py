"""
Code to quickly view a stellar spectrum

Process:
1) Extract target + comparison spectra from raw files
2) Divide it by a model A0 spectrum to get trend
3) Correct for this trend in the target spectrum
"""

import glob, os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import sigmaclip
warnings.filterwarnings("ignore", message="The following header keyword is invalid or follows an unrecognized "
                                          "non-standard convention")

nights = ['20170926', '20171009']       # dates of observations
ds = 1                                  # dataset to use

base_dir = nights[ds] + '/'             # folder of dataset

all_data = {}                           # holds data of each .fit file
obj_names, obj_use = [], []             # unique object names and object names to use
for fit_name in glob.glob(base_dir + '*.fit'):                                  # loop all .fit files
    f = fits.open(fit_name)                                                     # load data
    et_str = str(round(f[1].header['EXPTIME'], 1))
    obj = f[0].header['OBJECT'] + '_' + f[0].header['LIRGRNAM'][-2:] + '_' + et_str     # object name

    if obj not in obj_names:    # if first of object type -> add format
        all_data[obj] = {'filenames':[], 'raw_imgs':[], 'exp_times':[], 'master':[]}     # data to save

    all_data[obj]['filenames'] += [fit_name]                    # filename
    all_data[obj]['raw_imgs'] += [f[1].data]                    # raw pixel counts
    all_data[obj]['exp_times'] += [f[1].header['EXPTIME']]      # exposure time

    # print obj, f[1].header['EXPTIME'], '\t\t', fit_name
    # print all_data['DomeFlat-HK-Bright_hk']['exp_times']

    obj_names.append(obj)
    if 'acq' not in obj and obj not in obj_use:
        obj_use += [obj]

# for item in list(set(all_data)):
#     print item

"""
Comparison - A:HD27267_hk_6.0, B:HD27267_hk_6.0, A:HD27267_hk_3.0, B:HD27267_hk_3.0, A:HD27267_zj_1.5, B:HD27267_zj_1.5,
             A:HD27267_zj_4.0, B:HD27267_zj_4.0

Target - A:LP358_hk_12.0, B:LP358_hk_12.0, A:LP358_zj_25.0, B:LP358_zj_25.0

Arcs - Arc-Xe-short_hk_8.0, Arc-Xe-long_hk_80.0
"""

a0v_file = np.genfromtxt('uka0v.dat', delimiter='  ', dtype=float)
a0v_wav_full, a0v_spec_full = a0v_file[:,0] / 1e4, a0v_file[:,1]

hk_wav = [1.388, 2.419]     # microns
zj_wav = [0.887, 1.531]

a0v_wav, a0v_spec = [], []
for i in range(len(a0v_spec_full)):
    if hk_wav[0] < a0v_wav_full[i] < hk_wav[1] and i%2==0:
        a0v_wav.append(a0v_wav_full[i])
        a0v_spec.append(a0v_spec_full[i])

ab = [all_data['A:HD27267_hk_6.0']['raw_imgs'], all_data['B:HD27267_hk_6.0']['raw_imgs']]
all_rows = ab[0]
rows = all_rows[0]

# s_row, ap = 395, 5      # spectrum mean row, aperture
sum_vals = [[395, 5], [443, 5]]

print "> Combining", len(all_rows) * 2, "spectra."

spec = [0.0] * len(rows)    # summed spectrum
ab_ind = 0
for all_rows in ab:
    s_row, ap = sum_vals[ab_ind]
    for rows in all_rows:       # images
        for j in range(s_row-ap, s_row+ap+1):   # rows to sum
            for k in range(len(rows)):          # for each pixel in row, add to pixel counts
                spec[k] += rows[j][k]
    ab_ind += 1

spec = spec[7:]
wav_guess = np.linspace(hk_wav[0], hk_wav[1], len(spec))
a0v_spec /= np.nanmedian(a0v_spec)
spec /= np.nanmedian(spec)
spec -= 0.4
a0v_wav, a0v_spec = a0v_wav[:len(wav_guess)], a0v_spec[:len(wav_guess)]

print wav_guess[1]-wav_guess[0], a0v_wav[1]-a0v_wav[0], len(wav_guess), len(a0v_wav)

plt.plot(wav_guess, spec, label='HD27267')
plt.plot(wav_guess, a0v_spec, label='Model A0V')

earth_file = np.genfromtxt('mktrans_zm_10_10.dat')
earth_wav_full, earth_spec_full = earth_file[:,0], earth_file[:,1]
plt.plot(earth_wav_full, earth_spec_full, label='Earth', alpha=0.4)
plt.xlim(min(wav_guess)-0.1, max(wav_guess)+0.1)
plt.legend()
plt.xlabel('Wavelength (microns)')
plt.ylabel('Normalised flux')
plt.tight_layout()
plt.show()

trend = spec / a0v_spec
# plt.plot(wav_guess, trend)
# plt.show()

# - - - - - - - - - - - - - - - -

ab = [all_data['A:LP358_hk_12.0']['raw_imgs'], all_data['B:LP358_hk_12.0']['raw_imgs']]
all_rows = ab[1]
rows = all_rows[0]

sum_vals = [[401, 5], [449, 5]]

print "> Combining", len(all_rows) * 2, "spectra."

spec = [0.0] * len(rows)    # summed spectrum
ab_ind = 0
for all_rows in ab:
    s_row, ap = sum_vals[ab_ind]
    for rows in all_rows:       # images
        for j in range(s_row-ap, s_row+ap+1):   # rows to sum
            for k in range(len(rows)):          # for each pixel in row, add to pixel counts
                spec[k] += rows[j][k]
    ab_ind += 1

spec = spec[7:]
spec /= np.nanmedian(spec)
spec -= 0.4

spec_cor = spec/trend
spec_clip, wav_clip = [], []
for i in range(len(spec_cor)):
    if 0.0 < spec_cor[i] < 1.5:
        spec_clip.append(spec_cor[i])
        wav_clip.append(wav_guess[i])

print len(spec_cor) - len(spec_clip)

m1v_file = np.genfromtxt('ukm1v.dat', delimiter='  ', dtype=float)
m1v_wav_full, m1v_spec_full = m1v_file[:,0] / 1e4, m1v_file[:,1]

m1v_wav, m1v_spec = [], []
for i in range(len(m1v_spec_full)):
    if hk_wav[0] < m1v_wav_full[i] < hk_wav[1] and i%2==0:
        m1v_wav.append(m1v_wav_full[i])
        m1v_spec.append(m1v_spec_full[i])
m1v_spec /= np.nanmedian(m1v_spec)

# plt.plot(wav_guess, spec_cor)
# plt.plot(wav_clip, spec_clip)
# plt.plot(m1v_wav, m1v_spec)
# plt.plot(a0v_wav, a0v_wav)
# plt.plot(spec)
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('Normalised flux')
# plt.show()

# col = []
# for row in rows:
#     col.append(row[200])
# plt.plot(col)
# plt.show()

# plt.imshow(rows, origin='lower', vmin=100.0, vmax=600.0, cmap='gray')
# plt.colorbar()
# plt.show()
