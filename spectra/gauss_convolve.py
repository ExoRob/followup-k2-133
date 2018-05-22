from __future__ import division
from PyAstronomy import pyasl
import matplotlib.pylab as plt
import numpy as np
import pickle
import astropy.io.fits as fits
from pysynphot import observation
from pysynphot import spectrum


def load_model_spec(fname_spec, wav_lims, fname_wav='data/wavegrid.fits'):
    """
    Loads a stellar model spectrum into wavelength & intensity arrays
    :param fname_spec: Filename of spectrum to load
    :param wav_lims: Wavelength limits list - e.g. [1.0, 2.0]
    :param fname_wav: Filename of wavelength grid
    :return: Wavelength & intensity arrays
    """
    fm = fits.open(fname_spec)
    fw = fits.open(fname_wav)

    model_wav_full, model_spec_full = fw[0].data / 1e4, fm[0].data

    model_wav, model_spec= [], []
    for i in range(len(model_spec_full)):
        if wav_lims[0] <= model_wav_full[i] <= wav_lims[1]:
            model_wav.append(model_wav_full[i])
            model_spec.append(model_spec_full[i])

    return np.asarray(model_wav), np.asarray(model_spec) #/ np.nanmedian(model_spec)


def rebin_spec(wavin, specin, wavnew):
    """
    Filter spectrum to new wavelength grid
    :param wavin: Wavelength grid of spectrum
    :param specin: Spectrum to filter
    :param wavnew: New wavelength grid for spectrum
    :return: Spectrum for new wavelength grid
    """
    wavnew = np.asarray(wavnew)
    spec = spectrum.ArraySourceSpectrum(wave=wavin, flux=specin)
    dmy = np.ones(len(wavin))
    filt = spectrum.ArraySpectralElement(wavin, dmy, waveunits='micron')
    obs = observation.Observation(spec, filt, binset=wavnew, force=['taper', 'extrap'][1])

    return obs.binflux


with open('extract.pkl', 'rb') as pf:
    specs, wavs = pickle.load(pf)

# print wavs, specs

x = wavs['zj']
y = specs['zj']['comp']['both']

m_w, m_s = load_model_spec('data/9400_5.0.fits', [min(x), max(x)])

# # Apply Gaussian instrumental broadening, setting the resolution to 10000.
# r, fwhm1 = pyasl.instrBroadGaussFast(x, y, 700,
#           edgeHandling="firstlast", fullout=True)
#
# rbg = pyasl.broadGaussFast(x, y, 0.0017,
#           edgeHandling="firstlast")
#
# # Apply Gaussian instrumental broadening, setting the resolution to 10000.
# # Limit the extent of the Gaussian broadening kernel to five standard
# # deviations.

allwav = np.linspace(min(x), max(x), len(x))

allmodspec = rebin_spec(m_w, m_s, allwav)
allspec = rebin_spec(x, y, allwav)

# r = pyasl.broadGaussFast(allwav, allmodspec, 0.0025/2.355, edgeHandling="firstlast", maxsig=5.0)

r2, fwhm2 = pyasl.instrBroadGaussFast(allwav, allmodspec, 700, edgeHandling="firstlast", fullout=True, maxsig=5.0)

print(fwhm2)

# plt.plot(allwav, allspec, 'k-')
# plt.plot(allwav, r*25e2/max(r), 'r-', label="bgf")
# plt.plot(allwav, r2*25e2/max(r2), 'b-', label="ibgf")
# plt.legend(loc=4)
# plt.show()

trend = allspec / r2

# plt.plot(allwav, trend)
# plt.show()


alltarg = rebin_spec(x, specs['zj']['targ']['both'], allwav)
plt.plot(allwav[2:], alltarg[2:]/trend[2:], 'k-')
for teff in ['3400', '3600', '3800', '4000']:
    t_w, t_s = load_model_spec('data/'+teff+'_5.0.fits', [min(x), max(x)])
    allmodtarg = rebin_spec(t_w, t_s, allwav)

    rt, fwhmt = pyasl.instrBroadGaussFast(allwav, allmodtarg, 700, edgeHandling="firstlast", fullout=True, maxsig=5.0)

    plt.plot(allwav, rt/max(rt)*max(alltarg[2:5]/trend[2:5]))
plt.show()
