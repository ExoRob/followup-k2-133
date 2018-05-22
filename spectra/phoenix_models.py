import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from pysynphot import observation, spectrum

def rebin_spec(wavin, specin, wavnew, convolve=False):
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

    if convolve:
        spec_convolved = pyasl.instrBroadGaussFast(wavnew, obs.binflux, 700, edgeHandling="firstlast", maxsig=5.0)
        return spec_convolved

    else:
        return obs.binflux


def rebin_spec_convole_first(wavin, specin, wavnew):
    """
    MUST BE IN CONSTANT STEPS
    """
    wav_spaced = np.linspace(0.5, 2.5, 2e6, endpoint=False)
    spec_spaced = rebin_spec(wavin, specin, wav_spaced)
    spec_convolved = pyasl.instrBroadGaussFast(wav_spaced, spec_spaced, 700, edgeHandling="firstlast", maxsig=5.0)

    plt.plot(wav_spaced, spec_spaced)
    plt.plot(wav_spaced, spec_convolved)
    # plt.show()

    wavnew = np.asarray(wavnew)
    spec = spectrum.ArraySourceSpectrum(wave=wav_spaced, flux=spec_convolved)
    dmy = np.ones(len(wav_spaced))
    filt = spectrum.ArraySpectralElement(wav_spaced, dmy, waveunits='micron')
    obs = observation.Observation(spec, filt, binset=wavnew, force=['taper', 'extrap'][1])

    return obs.binflux


teff = '3600'

mfn = 'data/PHOENIX-ACES-AGSS-COND-2011_A1FITS_Z-0.0/lte0'+teff+'-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
hfn = 'data/'+teff+'_5.0.fits'
wfn = 'data/wavegrid.fits'
# ifn = 'data/lte0'+teff+'-5.00-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

fh = fits.open(hfn)
fm = fits.open(mfn)
fw = fits.open(wfn)
# fi = fits.open(ifn)

sh = fh[0].data
wh = fw[0].data / 1e4
sm = fm[0].data
wm = np.linspace(0.3, 10.0, len(sm))

# print fm[0].header
# print fh[0].header
# print fw[0].header


wc = np.linspace(0.8, 2.5, 2000)
sc = rebin_spec_convole_first(wh, sh, wc)

# sc /= np.nanmedian(sc)

# plt.plot(wh, sh)
plt.plot(wc, sc)
plt.show()

# plt.plot(wh, sh)
# plt.plot(wm, sm)
# plt.plot(wc, sc)
# plt.xlim(1.2, 2.4)
# plt.show()

# fig, axes = plt.subplots(1, 3, sharey=True)
#
# lims = [[1.1,1.4], [1.48,1.8], [2.0, 2.4]]
#
# for i in range(3):
#     ax, lim = axes[i], lims[i]
#
#     dw, ds = [], []
#     for j in range(len(wc)):
#         if lim[0] <= wc[j] <= lim[1]:
#             dw.append(wc[j])
#             ds.append(sc[j])
#     ds = np.asarray(ds) / np.median(ds)
#
#     ax.plot(dw, ds)
#     # ax.set_xlim(lim)
# plt.show()
