"""
Code to extract spectra from WHT/LIRIS

Process:
Note: Bias taken and subtracted for each exposure // Dark flats_fits not needed either.
1) Flat field correction - dome + tungsten lamp flats
2) Sky subtraction - nod pointings / near aperture
3) Curvature correction - not needed?
4) Wavelength calibration - Arc lamp spectra - Ar + Xe
5) Offset computation - header AUTOX/AUTOY / manual values
6) Spectra co-addition
7) Extraction
8) Telluric absorption correction - standard star

Target: LP 358-499 (M1V?)
Standard: HD 27267 (A0V)

To do:
- cosmic ray correction
- save values used - e.g. mean rows, apertures, wavelength-pixel solution
"""

import glob, os, sys, shutil, detect_peaks, pickle
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import medfilt as mf
from pysynphot import observation
from pysynphot import spectrum
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sys.path.append("/Users/rwells/Documents/posx/src/")
import stdextr as se
import Marsh, scipy
from PyAstronomy import pyasl


def set_warnings():
    warnings.filterwarnings("ignore", message="The following header keyword is invalid or follows an unrecognized "
                                              "non-standard convention")  # ignore lamp header warning
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")
    warnings.filterwarnings("error", message="Covariance of the parameters could not be estimated")
set_warnings()

def running_sigma_clip(data, ax, usig=3, lsig=3, binsize=50):
    #
    # Sigma clipping (running): find local outliers
    #
    data_clipped = []
    ax_clipped = []
    upperlist = []
    lowerlist = []
    i = 0
    while i < len(data):
        bin_begin = max(0, (i - binsize/2))
        bin_end = min(len(data),(i+binsize/2))
        the_bin = data[bin_begin:bin_end]

        std = np.nanstd(np.sort(the_bin)[1:])
        median = np.median(the_bin)
        upperbound = (median + (usig*std))
        lowerbound = (median - (lsig*std))
        upperlist.append(upperbound)
        lowerlist.append(lowerbound)
        if (data[i] < min([upperbound])) and (data[i] > max([lowerbound])) and (std < 10.0):
            data_clipped.append(data[i])
            ax_clipped.append(ax[i])

        i = i + 1

    return data_clipped, ax_clipped


def remove_sky(img, s_row, mask=None, ap=5, dis=10):

    # upper = img[s_row+dis:s_row+dis+ap]
    # lower = img[s_row-dis-ap:s_row-dis]

    both = np.append(img[s_row+dis:s_row+dis+ap], img[s_row-dis-ap:s_row-dis], axis=0)

    # ispc1 = np.nanmedian(upper.T, axis=1)
    # ispc2 = np.nanmedian(lower.T, axis=1)
    ispcb = np.nanmedian(both.T, axis=1)

    # print np.nanmedian(ispc1), np.nanmedian(ispc2), np.nanmedian(ispcb)

    # plt.plot(ispc1)
    # plt.plot(ispc2)
    # plt.plot(ispcb, color='k')
    # plt.show()

    return img - ispcb


def normalise(lis, method='median'):
    """
    Normalise array by specified method
    :param lis: Array to normalise
    :param method: Method to use: median, mean or max
    :return: Normalised array
    """
    if method == 'median':
        return np.asarray(lis) / np.nanmedian(lis)
    elif method == 'mean':
        return np.asarray(lis) / np.nanmean(lis)
    elif method == 'max':
        return np.asarray(lis) / np.nanmax(lis)
    else:
        raise AssertionError("Method not implemented'!", method)


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


def getSimpleSpectrum(data, trace_coeffs, Aperture, min_col, max_col):
    Result, ap = Marsh.SimpleExtraction((data.flatten()).astype('double'),
                                        scipy.polyval(trace_coeffs, np.arange(data.shape[1])).astype('double'),
                                        data.shape[0], data.shape[1],
                                        data.shape[1], Aperture, min_col, max_col)
    FinalMatrix = np.asarray(Result)  # After the function, we convert our list to a Numpy array.
    return FinalMatrix, ap

def PCoeff(data, trace_coeffs, Aperture, RON, Gain, NSigma, S, N, Marsh_alg, min_col, max_col):
    Result = Marsh.ObtainP((data.flatten()).astype('double'),
                           scipy.polyval(trace_coeffs, np.arange(data.shape[1])).astype('double'),
                           data.shape[0], data.shape[1], data.shape[1], Aperture, RON, Gain,
                           NSigma, S, N, Marsh_alg, min_col, max_col)
    FinalMatrix = np.asarray(Result)  # After the function, we convert our list to a Numpy array.
    FinalMatrix.resize(data.shape[0], data.shape[1])  # And return the array in matrix-form.
    return FinalMatrix

def getSpectrum(P, data, trace_coeffs, Aperture, RON, Gain, S, NCosmic, min_col, max_col):
    Result, size = Marsh.ObtainSpectrum((data.flatten()).astype('double'),
                                        scipy.polyval(trace_coeffs, np.arange(data.shape[1])).astype('double'),
                                        P.flatten().astype('double'), data.shape[0],
                                        data.shape[1], data.shape[1], Aperture, RON,
                                        Gain, S, NCosmic, min_col, max_col)
    FinalMatrix = np.asarray(Result)  # After the function, we convert our list to a Numpy array.
    FinalMatrix.resize(3, size)  # And return the array in matrix-form.
    return FinalMatrix

def lorentz(x, I, x_0, HWHM):
    return I * (HWHM ** 2 / ((x - x_0) ** 2 + HWHM ** 2))

def gaussian(x, I, x_0, sig):
    return I * (np.exp(-np.power(x - x_0, 2.) / (2 * np.power(sig, 2.))))


def sum_spectra(obj, s_row, mask, method='optimal', imgs='raw_imgs', ap=5, sub_sky=False, s_ap=5):
    """
    Sum spectra of object - multiple images & rows
    :param obj: Object name to sum
    :param s_row: Mean pixel row of spectrum
    :param mask: Bad pixel mask to use
    :param method: Extraction method to use - simple, box, gaussian, optimal
    :param imgs: Set of images to use
    :param ap: Aperture size - either side of mean row
    :return: Summed spectrum
    """
    n_imgs = len(all_data[obj][imgs])   # number of images to sum
    spec = np.zeros(n_pix)

    # just sum rows within aperture
    if method == 'simple':
        for i in range(n_imgs):
            if sub_sky:
                sky_rows = all_data[obj][imgs][i][s_row-ap-s_ap-1:s_row-ap-1] + \
                           all_data[obj][imgs][i][s_row+ap+1:s_row+ap+s_ap+1]
                sky_b = np.median(sky_rows)

            for j in range(s_row-ap, s_row+ap+1):   # rows to sum
                for k in range(n_pix):              # for each pixel in row, add to pixel counts
                    spec[k] += all_data[obj][imgs][i][j][k] / n_imgs    #/ (2.0 * ap + 1.0)
                    if sub_sky:
                        spec[k] -= sky_b / n_imgs

    # standard box extraction
    if method == 'box':
        for i in range(n_imgs):
            # input shape [nwavelength, nposition]
            ispc, isstd = se.stdextr(all_data[obj][imgs][i].T, s_row-ap, s_row+ap+1, mask=mask)
            spec += ispc
        spec /= n_imgs

    # fit gaussian / optimal extraction algorithm
    if method in ['gaussian', 'optimal']:
        for i in range(n_imgs):
            # img = np.ma.masked_array(data=all_data[obj][imgs][i], mask=mask, fill_value=0.0).filled()
            img = all_data[obj][imgs][i]
            slices = img[s_row - ap:s_row + ap + 1].T
            cpix, cents = [], []
            # trace = []
            for j in range(1024):
                col = slices[j]
                x = np.linspace(-ap, ap, len(col))

                try:
                    # popt, pcov = scipy.optimize.curve_fit(lorentz, x, col, [max(col), 0.0, 0.8])
                    popt, pcov = scipy.optimize.curve_fit(gaussian, x, col, [max(col), 0.0, 2.0])
                    cents.append(popt[1])
                    cpix.append(float(j))

                    if method == 'gaussian':
                        spec[j] += popt[0]

                    # col_rn = np.nanmax(slices[max([0,j-100]):min([1024,j+100])], axis=1)
                    # trace.append(float(np.argmin(abs(col_rn - np.nanmedian(col_rn)))))
                    # trace.append(np.nanmedian(col_rn))

                except (RuntimeError, scipy.optimize.OptimizeWarning, ValueError):
                    # cents.append(np.nan)
                    # cents.append(10.0)

                    if method == 'gaussian':
                        spec[j] = np.nan
                    pass

                # x_ss = np.linspace(-ap, ap, len(col)*100+1)
                # l_fit = lorentz(x_ss, popt[0], popt[1], popt[2])
                # # g_fit = gaussian(x_ss, popt[0], popt[1], popt[2])
                # plt.plot(x, col, 'o')
                # plt.plot(x_ss, l_fit)
                # # plt.plot(x_ss, g_fit)
                # plt.show()

            # y_c, x_c = running_sigma_clip(cents, np.arange(1024), binsize=50, usig=2.0, lsig=2.0)
            # z = np.polyfit(x_c, y_c, 4)
            # p = np.poly1d(z)
            #
            # plt.plot(x_c, y_c, '.')
            # plt.plot(p(np.arange(1024)))
            # plt.show()

            if method == 'optimal':
                ypix = np.asarray(cents) + s_row
                y_c, x_c = running_sigma_clip(ypix, cpix, binsize=300, usig=1.0, lsig=1.0)
                z = np.polyfit(x_c, y_c, 2)

                p = np.poly1d(z)
                plt.plot(x_c, y_c, '.')
                plt.plot(p(np.arange(1024)))
                plt.xlabel('Spectral pixel')
                plt.ylabel('Spatial pixel')
                plt.title('Trace for ' + obj + ' - image ' + str(i))
                plt.savefig(base_dir + 'out/' + obj+'_'+str(i)+'.png', format='png')
                plt.clf()
                # plt.show()

                enoise = 17.0                   # Read Out Noise of the measurements in electrons.
                egain = 4.0                     # Gain of the CCD in electrons/ADU.
                NSigma_P_Marsh = 5.0            # Number-of-sigma for the rejection of points in the Marsh algorithm.
                NSigma_Cosmic_Marsh = 100.0     # Same for the cosmic-ray rejection algorithm.
                S_Marsh = 0.4                   # Spacing between polynomials of the optimal extraction.
                N_Marsh = 3                     # Order of the polynomials to be fitted.
                Marsh_alg = 0                   # Type of algorithm for the Optimal Extraction (Marsh - curved)

                P_marsh = PCoeff(img, z, ap, enoise, egain, NSigma_P_Marsh, S_Marsh, N_Marsh, Marsh_alg, 0, 0)

                Sm = getSpectrum(P_marsh, img, z, ap, enoise, egain, S_Marsh, NSigma_Cosmic_Marsh, 0, 0)

                spec += Sm[1]

    return spec


def get_wav_sol(arc_row_ar, arc_row_xe, grism, mfn=21, tol=0.03, poly_ord=3,
                plot_arcs=False, plot_fit=False, plot_resid=False):
    """
    Obtain wavelength solution - pixel-wavelength
    :param arc_row_ar: Argon pixel spectrum
    :param arc_row_xe: Xenon pixel spectrum
    :param grism: Grism used, e.g. hk or zj
    :param mfn: Median filter window
    :param tol: Tolerance - max distance between guess and line
    :param poly_ord: Order of polynomial fit of solution
    :param plot_arcs: Plot arc spectra & line lists
    :param plot_fit: Plot wavelength solution
    :param plot_resid: Plot residuals to solution
    :return: Wavelength at each pixel
    """

    argon_wav = np.genfromtxt('data/ar_'+grism+'.lis')
    xenon_wav = np.genfromtxt('data/xe_'+grism+'.lis')

    if plot_arcs:
        fig, ax1 = plt.subplots()  # pixel position
        ax2 = ax1.twiny()  # wavelength
        ax1.plot(x_pix, arc_row_ar)
        ax1.plot(x_pix, arc_row_xe)
        ax1.set_xlabel('Pixel number')
        ax1.set_ylabel('Intensity')
        [ax2.axvline(aw, color='k') for aw in list(argon_wav) + list(xenon_wav)]  # line list
        ax2.set_xlabel('Wavelength (microns)')
        ax2.set_xlim(wav_ranges[grism][0], wav_ranges[grism][1])
        plt.show()

    # xpos = [41.0, 58.0, 81.0, 133.0, 155.0, 270.0, 304.0, 438.0, 585.0, 591.0, 599.0, 633.0, 649.0, 674.0, 724.0, 731.0,
    #         739.0, 761.0, 780.0, 789.0, 795.0, 855.0, 1007.0]
    # ang = [0.91255, 0.9227, 0.93568, 0.96604, 0.97872, 1.04729, 1.06765, 1.14912, 1.24062, 1.24427, 1.24911, 1.27057,
    #        1.28062, 1.29602, 1.32763, 1.33168, 1.33708, 1.35079, 1.36264, 1.36823, 1.37223, 1.40975, 1.50506]

    ar_mf = arc_row_ar - mf(arc_row_ar, mfn)
    xe_mf = arc_row_xe - mf(arc_row_xe, mfn)
    lines_both = [argon_wav, xenon_wav]
    mf_both = [ar_mf, xe_mf]

    z = np.polyfit(x_pix, np.linspace(wav_ranges[grism][0], wav_ranges[grism][1], 1024), 1)
    sol = np.poly1d(z)

    # plt.plot(xpos, ang, '.', color='k')
    # plt.plot(x_pix, sol(x_pix), color='k', ls='--')

    x_g, w_g, l_g, dif = [], [], [], []
    for arc in range(2):  # each arc lamp
        xpos = list(detect_peaks.detect_peaks(mf_both[arc], mpd=7, threshold=150.0))  # pixel numbers of peaks
        lines = lines_both[arc]  # line list for lamp

        for i in range(len(xpos)):      # loop x positions of lines
            pix = xpos[i]           # pixel number
            wav_guess = sol(pix)    # guess arc line
            li = np.argmin(abs(np.asarray(lines) - wav_guess))  # closest line to guess
            line_i = lines[li]
            dif_i = abs(wav_guess - lines[li])  # wavelength difference between guess and matched line

            if dif_i < tol and int(pix) not in [512, 513, 743, 948]:  # if match close to guess & not a bad pixel
                done = line_i in l_g  # if line already matched to a pixel
                if done:
                    j = np.argmin(abs(np.asarray(w_g) - wav_guess))  # index in x_g, w_g, l_g, dif
                    dif_j = dif[j]
                    print line_i, l_g[j], dif_i, dif_j, pix, x_g[j]

                    if dif_i < dif_j:  # if new pixel closer
                        del x_g[j]
                        del w_g[j]
                        del l_g[j]
                        del dif[j]

                        x_g.append(pix)  # pixel number
                        w_g.append(wav_guess)  # guess value
                        l_g.append(line_i)  # line value matched
                        dif.append(dif_i)  # difference between guess and line matched

                else:
                    x_g.append(pix)  # pixel number
                    w_g.append(wav_guess)  # guess value
                    l_g.append(line_i)  # line value matched
                    dif.append(dif_i)  # difference between guess and line matched

        if plot_fit:
            [plt.axhline(aw, color='k') for aw in lines]
            [plt.axvline(aw, color='k') for aw in xpos]

    z = np.polyfit(x_g, l_g, poly_ord)
    sol = np.poly1d(z)
    xglg = sol(x_pix)

    print list(xglg)

    if plot_fit:
        plt.plot(x_pix, xglg, color='#ff7f0e', alpha=0.7, label='Fit')
        plt.plot(x_g, l_g, 'x', color='k')
        plt.xlabel('Pixel number')
        plt.ylabel('Wavelength (microns)')
        plt.legend()
        plt.show()

    if plot_resid:
        fitdif = []
        for i in range(len(x_g)):
            x = int(x_g[i])
            fit = sol(x)
            y = l_g[i]
            fitdif.append(y - fit)

        plt.plot(x_g, fitdif, 'o')
        plt.axhline(0, color='k')
        plt.show()

    return xglg


def get_mean_row(obj, lb=None, ub=None, plot_mean=False, plot_slices=False):
    """
    Get row number of slit centre
    :param obj: Object name
    :param lb: Lower bound
    :param ub: Upper bound
    :param plot_mean: Plot the mean row
    :param plot_slices: Plot slices across mean row -> aperture size
    :return: Index of mean row
    """
    rows = all_data[obj]['raw_imgs'][0]     # list of rows
    cols = rows.T                           # transpose to list of columns
    col_max = []                            # index of max of each column
    for j in range(n_pix):
        col_max.append(np.argmax(cols[j][lb:ub]))   # index of max, i.e. row number in column

    counts = np.bincount(col_max)   # count number of maxes
    s_row = np.argmax(counts)       # mean row = most common index

    if plot_mean:
        plt.plot(x_pix, col_max, '.')
        plt.axhline(s_row, ls='--', color='k')
        plt.show()

    if plot_slices:
        for j in range(200, 1001, 200):
            col = cols[j]
            i1, i2 = s_row-20, s_row+20
            plt.plot(x_pix[i1:i2], col[i1:i2], label=j)
        plt.axvline(s_row, ls='--', color='k')
        plt.legend()
        plt.show()

    return s_row


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


def plot_im(img, lims=None):
    if lims:
        l1, l2 = lims
    else:
        l1, l2 = np.nanmin(img), np.nanmax(img)

    plt.imshow(img, origin='lower', vmin=l1, vmax=l2, cmap='gray')
    plt.colorbar()
    plt.grid(True)
    plt.show()


def plot_transmission(plot_spectra=False, factor=1.0):
    if plot_spectra:
        earth_file = np.genfromtxt('data/mktrans_zm_10_10.dat')
        earth_wav_full, earth_spec_full = earth_file[:,0], factor * earth_file[:,1]
        earth_s_bin = rebin_spec(earth_wav_full, earth_spec_full, np.linspace(0.9, 2.4, 1650))

        plt.plot(earth_wav_full, earth_spec_full, label='Earth', alpha=0.4)
        plt.plot(np.linspace(0.9, 2.4, 1650), earth_s_bin, label='Earth', alpha=0.4, color='g')

    else:
        regions = [[0.93, 0.97], [1.11, 1.16], [1.31, 1.51], [1.72, 2.08], [2.25, 2.50]]
        for x1,x2 in regions:
            plt.axvspan(x1, x2, alpha=0.2, color='grey')

    plt.xlim(0.8, 2.5)
    # plt.show()


def plot_throughput(factor=1.0):
    for filt in ['z', 'j', 'h', 'k']:
        fwav, ftrans = np.genfromtxt('data/cold_'+filt+'.txt').T
        plt.plot(fwav, ftrans*factor)
    # plt.show()


# - - - - - - - -

nights = ['20170926', '20171009']       # dates of observations
ds = 1                                  # dataset to use
base_dir = nights[ds] + '/'             # folder of dataset

do_extraction = True

wav_ranges = {'hk':[1.388, 2.419], 'zj':[0.887, 1.531]}     # grism ranges in microns
n_pix = 1024            # number of pixels, width & height
x_pix = range(n_pix)    # pixel number along spectrum

print "> Loading data ..."

all_data = {}                           # holds data of each .fit file
obj_names, obj_use = [], []             # unique object names and object names to use
# pr = ''
# gains, rnoises = [], []
for fit_name in glob.glob(base_dir + '*.fit'):      # loop all .fit files
    f = fits.open(fit_name)                         # load data
    et_str = str(round(f[1].header['EXPTIME'], 1))                                      # exposure time
    obj = f[0].header['OBJECT'] + '_' + f[0].header['LIRGRNAM'][-2:] + '_' + et_str     # object name

    if obj not in obj_names:    # if first of object type -> add format
        all_data[obj] = {'filenames':[], 'raw_imgs':[], 'exp_times':[], 'master':[],    # data to save
                         'flat_cor_imgs':[]}

    all_data[obj]['filenames'] += [fit_name]                    # filename
    all_data[obj]['raw_imgs'] += [f[1].data]                    # raw pixel counts
    all_data[obj]['exp_times'] += [f[1].header['EXPTIME']]      # exposure time

    # pr += fit_name.split('/')[1]+','
    # print fit_name.split('/')[1]+'[1]'

    # print obj, f[1].header['EXPTIME'], '\t\t', fit_name
    # print all_data['DomeFlat-HK-Bright_hk']['exp_times']
    # gains.append(f[1].header['GAIN'])
    # rnoises.append(f[1].header['READNOIS'])

    obj_names.append(obj)
    if 'acq' not in obj and obj not in obj_use:
        obj_use += [obj]
        # print obj

# print pr
# print set(gains), '\n', set(rnoises)

# for thing in sorted(set(obj_names)):   # sorted(obj_use):
#     print thing, '-', len(all_data[thing]['raw_imgs'])

# dmy1, dmy2 = [], []
# for key in all_data.keys():
#     dmy1 += [len(set(all_data[key]['exp_times']))]
#     dmy2 += [len(all_data[key]['exp_times'])]
# print dmy1, '\n', dmy2

if do_extraction:
    # create master flats and arcs
    print "> Creating master flats and arcs ..."
    for obj in obj_use:                         # loop each object + grism
        if 'Flat' in obj or 'flat' in obj:      # disregard first 5 flat exposures
            all_data[obj]['raw_imgs'] = all_data[obj]['raw_imgs'][5:]
        obj_data = all_data[obj]                # all data for object + grism
        n_imgs = len(obj_data['raw_imgs'])      # number of .fit files
        exp_t_set = set(obj_data['exp_times'])

        if len(exp_t_set) == 1 and ('Ar' in obj or 'Xe' in obj or 'Flat' in obj or 'flat' in obj) and n_imgs > 0:
            # if single exposure time
            # print "> Creating master", obj, 'from', n_imgs, 'images ...'

            exp_t = obj_data['exp_times'][0]        # exposure time in seconds
            stack = []
            for img in obj_data['raw_imgs']:
                stack.append(img)
            stack = np.dstack(stack)                # stack all images
            med_stack = np.median(stack, axis=2)    # create median image
            all_data[obj]['master'] = med_stack

            master_obj = fits.PrimaryHDU(med_stack)
            header = {'OBJECT':obj+'_master', 'EXPTIME':exp_t}
            master_obj.writeto(base_dir + 'out/' + obj + '_master.fit', overwrite=True)     # save to master .fit file

        # elif n_imgs == 1:       # if only one image -> copy to master .fit file
        #     print "Copying single file to master", obj, "..."
        #     shutil.copy(obj_data['filenames'][0], base_dir + 'out/' + obj + '_master.fit')

        elif len(exp_t_set) > 1:   # if multiple exposure times used -> normalise and combine
            print "\n>", obj, "has", n_imgs, "files.", set(obj_data['exp_times'])
            sys.exit()


    # create bad pixel mask from flats
    if ds == 1:
        bri = all_data['DomeFlat-HK-Bright_hk_3.0']['master']   # bright flat -> dead pixels
        dim = all_data['DomeFlat-HK-Dim_hk_3.0']['master']      # dim flat -> hot pixels
    elif ds == 0:
        bri = all_data['bright spec flat_hk_1.5']['master']
        dim = all_data['dim spec flat_hk_1.5']['master']

    # hp = np.where(dim > 5.0 * np.median(dim))       # hot pixel location - row-column
    hp_mask = (dim > 5.0 * np.median(dim))          # hot pixel mask
    # print len(hp[0]), 'hot pixels.'
    # for i in range(len(hp[0])):
    #     r, c = hp[1][i], hp[0][i]
    #     plt.plot(r, c, '.', color='r')
    # plot_im(dim)

    # cp = np.where(bri <= 0.15 * np.median(bri))     # cold pixel location - row-column
    cp_mask = (bri <= 0.15 * np.median(bri))        # cold pixel mask
    # print len(cp[0]), 'cold pixels.'
    # for i in range(len(cp[0])):
    #     r, c = cp[1][i], cp[0][i]
    #     plt.plot(r, c, '.', color='r')
    # plot_im(bri)

    grid = np.meshgrid(np.arange(1024), np.arange(1024))[0]
    cc_mask = (grid == 512) | (grid == 513) | (grid == 514)         # chip crossover mask

    bad_pixel_mask = (hp_mask == 1) | (cp_mask == 1) | (cc_mask == 1)       # combine all bad pixel masks
    print "> Identified", np.count_nonzero(bad_pixel_mask), "bad pixels."

    # for obj in obj_use:
    #     if 'master' in all_data[obj].keys():
    #         if len(all_data[obj]['master']) > 0:
    #             all_data[obj]['master'] = np.ma.masked_array(data=all_data[obj]['master'], mask=bad_pixel_mask)


    # print "\n> Subtracting dims from brights.\n"
    # masters = {}    # bright - dim
    # for obj in obj_use:
    #     if 'Bright' in obj and 'zJ' not in obj:     # Bright and Dim flats
    #         sub = obj.replace('-Bright', '')
    #         obj2 = obj.replace('Bright', 'Dim')
    #         masters[sub] = all_data[obj]['master'] - all_data[obj2]['master']
    #
    #     elif 'long' in obj:     # Long and Short arcs
    #         sub = obj.replace('-long', '')
    #         obj2 = obj.replace('long', 'short')
    #         masters[sub] = all_data[obj]['master'] - all_data[obj2]['master']
    #
    #     elif 'Dim' in obj or 'short' in obj:    # covered by bright/long
    #         pass    # skip
    #
    #     else:       # no subtraction needed
    #         masters[obj] = all_data[obj]['master']
    # print masters.keys()

    if ds == 1:
        DomeFlat_hk = all_data['DomeFlat-HK-Bright_hk_3.0']['master'] - all_data['DomeFlat-HK-Dim_hk_3.0']['master']
        # DomeFlat_zj = all_data['DomeFlat-zJ-Bright_zj']['master']
        WFlat_zj = all_data['W-Flat-zJ-Bright_zj_1.5']['master']
        ArcAr_hk = all_data['Arc-Ar-long_hk_80.0']['master'] - all_data['Arc-Ar-short_hk_10.0']['master']
        ArcAr_zj = all_data['Arc-Ar-long_zj_15.0']['master'] - all_data['Arc-Ar-short_zj_3.5']['master']
        ArcXe_hk = all_data['Arc-Xe-long_hk_80.0']['master'] - all_data['Arc-Xe-short_hk_8.0']['master']
        ArcXe_zj = all_data['Arc-Xe-long_zj_60.0']['master'] - all_data['Arc-Xe-short_zj_6.0']['master']
        # ArcArXe_hk = all_data['Arc-Ar-Xe_hk']['master']
        # ArcArXe_zj = all_data['Arc-Ar-Xe_zj']['master']

    elif ds == 0:
        DomeFlat_hk = all_data['bright spec flat_hk_1.5']['master'] - all_data['dim spec flat_hk_1.5']['master']
        WFlat_zj = all_data['spec flat_zj_1.0']['master']
        ArcAr_hk = all_data['Ar long_hk_80.0']['master'] - all_data['Ar short_hk_9.4']['master']
        ArcAr_zj = all_data['Ar long_zj_13.4']['master'] - all_data['Ar short_zj_3.4']['master']
        ArcXe_hk = all_data['Xe long_hk_80.0']['master'] - all_data['Xe short_hk_6.7']['master']
        ArcXe_zj = all_data['Xe long_zj_54.0']['master']

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # [grism][star]
    if ds == 1:
        data = {'hk':{'comp':':HD27267_hk_6.0', 'targ':':LP358_hk_12.0'},
                'zj':{'comp':':HD27267_zj_4.0', 'targ':':LP358_zj_25.0'}}
    elif ds == 0:
        data = {'hk':{'comp':':HD 284647 HK_hk_8.0', 'targ':':LP358 HK_hk_25.0'},
                'zj':{'comp':':HD 284647 zJ_zj_10.0', 'targ':':LP358 zJ_zj_25.0'}}

    specs = {'hk':{'comp':{}, 'targ':{}}, 'zj':{'comp':{}, 'targ':{}}}
    wavs = {}

    print "> Flat-fielding and extracting spectra ..."

    # grism -> star -> nod point
    for gr in ['zj', 'hk']:     # each grism
        # create flat and apply to arcs
        if gr == 'zj':
            flat = np.ma.masked_array(data=WFlat_zj / np.nanmedian(WFlat_zj), mask=bad_pixel_mask)
            arc_ar = ArcAr_zj / flat
            arc_xe = ArcXe_zj / flat
            wv = np.asarray([0.88924291866864891, 0.88981189059590982, 0.89038100315572555, 0.89095025618071999, 0.89151964950351736, 0.89208918295674133, 0.89265885637301612, 0.89322866958496561, 0.8937986224252138, 0.89436871472638468, 0.89493894632110216, 0.89550931704199022, 0.89607982672167286, 0.89665047519277408, 0.89722126228791776, 0.89779218783972803, 0.89836325168082876, 0.89893445364384394, 0.89950579356139748, 0.90007727126611359, 0.90064888659061593, 0.90122063936752872, 0.90179252942947585, 0.90236455660908121, 0.90293672073896891, 0.90350902165176294, 0.90408145918008709, 0.90465403315656556, 0.90522674341382214, 0.90579958978448083, 0.90637257210116573, 0.90694569019650073, 0.90751894390310983, 0.90809233305361692, 0.9086658574806461, 0.90923951701682137, 0.90981331149476652, 0.91038724074710564, 0.91096130460646274, 0.9115355029054617, 0.91210983547672664, 0.91268430215288143, 0.91325890276654997, 0.91383363715035637, 0.91440850513692451, 0.9149835065588785, 0.91555864124884223, 0.91613390903943959, 0.91670930976329468, 0.9172848432530315, 0.91786050934127394, 0.9184363078606459, 0.91901223864377157, 0.91958830152327475, 0.92016449633177944, 0.92074082290190962, 0.92131728106628941, 0.92189387065754258, 0.92247059150829325, 0.92304744345116529, 0.92362442631878272, 0.92420153994376952, 0.9247787841587497, 0.92535615879634725, 0.92593366368918595, 0.92651129866989002, 0.92708906357108334, 0.9276669582253898, 0.92824498246543352, 0.92882313612383838, 0.92940141903322837, 0.9299798310262275, 0.93055837193545976, 0.93113704159354915, 0.93171583983311945, 0.93229476648679477, 0.93287382138719921, 0.93345300436695655, 0.93403231525869079, 0.93461175389502604, 0.93519132010858619, 0.93577101373199512, 0.93635083459787694, 0.93693078253885564, 0.93751085738755502, 0.93809105897659928, 0.93867138713861231, 0.939251841706218, 0.93983242251204036, 0.94041312938870347, 0.94099396216883113, 0.94157492068504756, 0.94215600476997641, 0.94273721425624202, 0.94331854897646805, 0.94390000876327862, 0.94448159344929772, 0.94506330286714935, 0.94564513684945739, 0.94622709522884585, 0.94680917783793872, 0.94739138450935989, 0.94797371507573358, 0.94855616936968346, 0.94913874722383362, 0.94972144847080819, 0.95030427294323094, 0.95088722047372587, 0.95147029089491708, 0.95205348403942847, 0.95263679973988402, 0.95322023782890763, 0.95380379813912342, 0.95438748050315525, 0.95497128475362714, 0.95555521072316307, 0.95613925824438695, 0.95672342714992287, 0.95730771727239472, 0.9578921284444265, 0.95847666049864222, 0.95906131326766586, 0.95964608658412132, 0.96023098028063258, 0.96081599418982366, 0.96140112814431855, 0.96198638197674113, 0.96257175551971552, 0.9631572486058656, 0.96374286106781537, 0.96432859273818872, 0.96491444344960975, 0.96550041303470246, 0.96608650132609064, 0.9666727081563985, 0.96725903335824981, 0.96784547676426869, 0.96843203820707902, 0.9690187175193048, 0.96960551453357002, 0.9701924290824987, 0.9707794609987147, 0.97136661011484204, 0.97195387626350482, 0.97254125927732682, 0.97312875898893214, 0.97371637523094479, 0.97430410783598864, 0.9748919566366876, 0.97547992146566587, 0.97606800215554723, 0.9766561985389558, 0.97724451044851546, 0.9778329377168502, 0.97842148017658404, 0.97901013766034084, 0.97959891000074473, 0.98018779703041958, 0.98077679858198941, 0.98136591448807808, 0.98195514458130972, 0.98254448869430833, 0.98313394665969778, 0.98372351831010207, 0.9843132034781451, 0.98490300199645098, 0.98549291369764369, 0.98608293841434702, 0.98667307597918519, 0.98726332622478197, 0.98785368898376147, 0.98844416408874758, 0.9890347513723643, 0.98962545066723562, 0.99021626180598554, 0.99080718462123796, 0.99139821894561686, 0.99198936461174636, 0.99258062145225023, 0.9931719892997527, 0.99376346798687742, 0.99435505734624863, 0.9949467572104902, 0.99553856741222613, 0.99613048778408031, 0.99672251815867685, 0.99731465836863964, 0.99790690824659267, 0.99849926762515995, 0.99909173633696546, 0.99968431421463311, 1.0002770010907869, 1.0008697967980509, 1.0014627011690487, 1.0020557140364048, 1.002648835232743, 1.0032420645906872, 1.0038354019428612, 1.0044288471218894, 1.0050223999603956, 1.0056160602910036, 1.0062098279463374, 1.0068037027590213, 1.0073976845616788, 1.0079917731869343, 1.0085859684674117, 1.0091802702357346, 1.0097746783245274, 1.010369192566414, 1.0109638127940181, 1.0115585388399639, 1.0121533705368755, 1.0127483077173767, 1.0133433502140916, 1.013938497859644, 1.0145337504866578, 1.0151291079277573, 1.0157245700155662, 1.0163201365827086, 1.0169158074618085, 1.0175115824854899, 1.0181074614863765, 1.0187034442970928, 1.0192995307502621, 1.0198957206785089, 1.0204920139144571, 1.0210884102907305, 1.021684909639953, 1.0222815117947488, 1.0228782165877419, 1.0234750238515562, 1.0240719334188155, 1.0246689451221442, 1.0252660587941658, 1.0258632742675045, 1.0264605913747844, 1.0270580099486293, 1.027655529821663, 1.0282531508265098, 1.0288508727957937, 1.0294486955621382, 1.0300466189581678, 1.0306446428165064, 1.0312427669697777, 1.0318409912506059, 1.0324393154916149, 1.0330377395254289, 1.0336362631846714, 1.0342348863019666, 1.0348336087099388, 1.0354324302412115, 1.0360313507284087, 1.0366303700041548, 1.0372294879010733, 1.0378287042517886, 1.0384280188889243, 1.0390274316451047, 1.0396269423529534, 1.0402265508450947, 1.0408262569541524, 1.0414260605127508, 1.0420259613535134, 1.0426259593090643, 1.0432260542120277, 1.0438262458950274, 1.0444265341906875, 1.0450269189316317, 1.0456273999504844, 1.0462279770798693, 1.0468286501524102, 1.0474294190007316, 1.0480302834574569, 1.0486312433552105, 1.049232298526616, 1.0498334488042977, 1.0504346940208797, 1.0510360340089855, 1.0516374686012395, 1.0522389976302653, 1.0528406209286871, 1.0534423383291289, 1.0540441496642146, 1.0546460547665684, 1.0552480534688138, 1.0558501456035749, 1.0564523310034761, 1.0570546095011411, 1.0576569809291938, 1.0582594451202583, 1.0588620019069586, 1.0594646511219183, 1.0600673925977619, 1.060670226167113, 1.0612731516625957, 1.0618761689168343, 1.0624792777624523, 1.0630824780320738, 1.0636857695583228, 1.0642891521738234, 1.0648926257111992, 1.0654961900030748, 1.0660998448820738, 1.0667035901808199, 1.0673074257319377, 1.0679113513680507, 1.0685153669217831, 1.0691194722257586, 1.0697236671126016, 1.0703279514149358, 1.0709323249653853, 1.0715367875965738, 1.0721413391411254, 1.0727459794316645, 1.0733507083008145, 1.0739555255811997, 1.0745604311054437, 1.075165424706171, 1.0757705062160055, 1.0763756754675706, 1.0769809322934909, 1.0775862765263902, 1.0781917079988923, 1.0787972265436214, 1.0794028319932014, 1.0800085241802562, 1.0806143029374098, 1.0812201680972864, 1.0818261194925096, 1.0824321569557034, 1.0830382803194922, 1.0836444894164996, 1.0842507840793496, 1.0848571641406664, 1.0854636294330737, 1.0860701797891958, 1.0866768150416564, 1.0872835350230794, 1.087890339566089, 1.0884972285033092, 1.089104201667364, 1.0897112588908771, 1.0903184000064725, 1.0909256248467745, 1.0915329332444068, 1.0921403250319934, 1.0927478000421584, 1.0933553581075257, 1.0939629990607194, 1.0945707227343633, 1.0951785289610814, 1.0957864175734977, 1.0963943884042362, 1.0970024412859209, 1.0976105760511756, 1.0982187925326246, 1.0988270905628916, 1.0994354699746007, 1.1000439306003758, 1.1006524722728408, 1.10126109482462, 1.101869798088337, 1.1024785818966161, 1.103087446082081, 1.1036963904773558, 1.1043054149150644, 1.1049145192278309, 1.1055237032482792, 1.1061329668090332, 1.1067423097427171, 1.1073517318819548, 1.10796123305937, 1.1085708131075871, 1.1091804718592297, 1.1097902091469221, 1.1104000248032879, 1.1110099186609514, 1.1116198905525365, 1.1122299403106672, 1.1128400677679673, 1.113450272757061, 1.114060555110572, 1.1146709146611247, 1.1152813512413426, 1.1158918646838498, 1.1165024548212708, 1.1171131214862287, 1.1177238645113481, 1.1183346837292527, 1.1189455789725666, 1.119556550073914, 1.1201675968659184, 1.120778719181204, 1.1213899168523946, 1.1220011897121147, 1.1226125375929876, 1.1232239603276377, 1.1238354577486891, 1.1244470296887652, 1.1250586759804906, 1.1256703964564889, 1.1262821909493841, 1.1268940592918002, 1.1275060013163614, 1.1281180168556915, 1.1287301057424144, 1.129342267809154, 1.1299545028885345, 1.1305668108131799, 1.1311791914157141, 1.1317916445287608, 1.1324041699849445, 1.1330167676168887, 1.1336294372572178, 1.1342421787385555, 1.1348549918935256, 1.1354678765547526, 1.1360808325548599, 1.136693859726472, 1.1373069579022124, 1.1379201269147052, 1.1385333665965749, 1.1391466767804448, 1.1397600572989393, 1.1403735079846817, 1.140987028670297, 1.1416006191884085, 1.1422142793716403, 1.1428280090526164, 1.1434418080639608, 1.1440556762382974, 1.1446696134082504, 1.1452836194064435, 1.1458976940655008, 1.1465118372180461, 1.1471260486967036, 1.1477403283340972, 1.1483546759628509, 1.1489690914155888, 1.1495835745249345, 1.1501981251235125, 1.1508127430439463, 1.1514274281188599, 1.1520421801808776, 1.1526569990626232, 1.1532718845967207, 1.1538868366157939, 1.1545018549524673, 1.1551169394393641, 1.155732089909109, 1.1563473061943255, 1.1569625881276377, 1.1575779355416698, 1.1581933482690454, 1.1588088261423886, 1.1594243689943236, 1.160039976657474, 1.1606556489644642, 1.161271385747918, 1.1618871868404592, 1.1625030520747119, 1.1631189812833, 1.1637349742988476, 1.1643510309539788, 1.1649671510813173, 1.1655833345134874, 1.1661995810831125, 1.1668158906228172, 1.1674322629652252, 1.1680486979429603, 1.1686651953886469, 1.1692817551349086, 1.1698983770143694, 1.1705150608596535, 1.1711318065033849, 1.1717486137781874, 1.1723654825166849, 1.1729824125515014, 1.1735994037152611, 1.1742164558405879, 1.1748335687601057, 1.1754507423064382, 1.1760679763122099, 1.1766852706100446, 1.1773026250325662, 1.1779200394123985, 1.1785375135821661, 1.1791550473744921, 1.179772640622001, 1.1803902931573167, 1.1810080048130633, 1.1816257754218646, 1.1822436048163447, 1.1828614928291272, 1.1834794392928365, 1.1840974440400966, 1.1847155069035313, 1.1853336277157644, 1.1859518063094203, 1.1865700425171226, 1.1871883361714954, 1.1878066871051627, 1.1884250951507485, 1.1890435601408766, 1.1896620819081714, 1.1902806602852565, 1.1908992951047559, 1.1915179861992937, 1.1921367334014938, 1.1927555365439801, 1.1933743954593767, 1.1939933099803077, 1.1946122799393968, 1.1952313051692682, 1.1958503855025457, 1.1964695207718534, 1.1970887108098152, 1.1977079554490551, 1.198327254522197, 1.1989466078618651, 1.1995660153006831, 1.2001854766712752, 1.2008049918062651, 1.201424560538277, 1.2020441826999351, 1.2026638581238627, 1.2032835866426843, 1.2039033680890239, 1.2045232022955052, 1.2051430890947523, 1.2057630283193892, 1.20638301980204, 1.2070030633753284, 1.2076231588718784, 1.2082433061243141, 1.2088635049652596, 1.2094837552273385, 1.2101040567431753, 1.2107244093453935, 1.2113448128666173, 1.2119652671394707, 1.2125857719965774, 1.2132063272705618, 1.2138269327940474, 1.2144475883996586, 1.2150682939200192, 1.2156890491877532, 1.2163098540354844, 1.2169307082958372, 1.2175516118014351, 1.2181725643849024, 1.2187935658788629, 1.2194146161159405, 1.2200357149287595, 1.2206568621499436, 1.2212780576121169, 1.2218993011479033, 1.2225205925899267, 1.2231419317708114, 1.2237633185231811, 1.2243847526796596, 1.2250062340728713, 1.2256277625354401, 1.2262493378999899, 1.2268709599991443, 1.2274926286655279, 1.2281143437317641, 1.2287361050304775, 1.2293579123942915, 1.2299797656558304, 1.230601664647718, 1.2312236092025786, 1.2318455991530355, 1.2324676343317136, 1.2330897145712361, 1.2337118397042273, 1.2343340095633111, 1.2349562239811116, 1.2355784827902525, 1.2362007858233581, 1.2368231329130523, 1.2374455238919591, 1.2380679585927021, 1.2386904368479059, 1.2393129584901939, 1.2399355233521905, 1.2405581312665193, 1.2411807820658045, 1.2418034755826701, 1.24242621164974, 1.2430489900996382, 1.2436718107649885, 1.2442946734784153, 1.2449175780725421, 1.2455405243799933, 1.2461635122333925, 1.2467865414653638, 1.2474096119085314, 1.2480327233955191, 1.2486558757589505, 1.2492790688314503, 1.2499023024456422, 1.2505255764341499, 1.2511488906295976, 1.2517722448646091, 1.2523956389718087, 1.2530190727838202, 1.2536425461332676, 1.2542660588527748, 1.2548896107749656, 1.2555132017324646, 1.2561368315578951, 1.2567605000838813, 1.2573842071430472, 1.258007952568017, 1.2586317361914146, 1.2592555578458635, 1.2598794173639882, 1.2605033145784124, 1.2611272493217602, 1.2617512214266555, 1.2623752307257226, 1.262999277051585, 1.2636233602368667, 1.2642474801141921, 1.2648716365161849, 1.265495829275469, 1.2661200582246686, 1.2667443231964073, 1.2673686240233095, 1.2679929605379989, 1.2686173325730996, 1.2692417399612359, 1.269866182535031, 1.2704906601271095, 1.2711151725700951, 1.2717397196966118, 1.2723643013392836, 1.2729889173307347, 1.2736135675035887, 1.2742382516904698, 1.2748629697240021, 1.275487721436809, 1.2761125066615153, 1.2767373252307443, 1.2773621769771204, 1.2779870617332674, 1.2786119793318091, 1.2792369296053696, 1.2798619123865733, 1.2804869275080435, 1.2811119748024045, 1.2817370541022803, 1.2823621652402948, 1.2829873080490721, 1.283612482361236, 1.2842376880094106, 1.2848629248262198, 1.2854881926442876, 1.286113491296238, 1.286738820614695, 1.2873641804322826, 1.2879895705816244, 1.2886149908953448, 1.2892404412060678, 1.2898659213464172, 1.290491431149017, 1.2911169704464913, 1.2917425390714636, 1.2923681368565587, 1.2929937636343998, 1.2936194192376111, 1.2942451034988169, 1.2948708162506408, 1.2954965573257069, 1.2961223265566393, 1.2967481237760619, 1.2973739488165985, 1.2979998015108731, 1.29862568169151, 1.2992515891911329, 1.2998775238423659, 1.3005034854778328, 1.3011294739301578, 1.3017554890319647, 1.3023815306158775, 1.3030075985145202, 1.3036336925605168, 1.3042598125864913, 1.3048859584250676, 1.3055121299088697, 1.3061383268705216, 1.3067645491426474, 1.3073907965578708, 1.308017068948816, 1.3086433661481067, 1.3092696879883672, 1.3098960343022215, 1.3105224049222932, 1.3111487996812066, 1.3117752184115854, 1.3124016609460538, 1.3130281271172359, 1.3136546167577552, 1.3142811297002364, 1.3149076657773027, 1.3155342248215784, 1.3161608066656876, 1.3167874111422542, 1.3174140380839021, 1.3180406873232553, 1.3186673586929378, 1.3192940520255736, 1.3199207671537865, 1.3205475039102008, 1.3211742621274403, 1.3218010416381287, 1.3224278422748905, 1.3230546638703493, 1.3236815062571294, 1.3243083692678543, 1.3249352527351486, 1.3255621564916356, 1.3261890803699397, 1.3268160242026847, 1.3274429878224947, 1.3280699710619936, 1.3286969737538055, 1.3293239957305543, 1.3299510368248637, 1.3305780968693581, 1.3312051756966614, 1.3318322731393972, 1.3324593890301899, 1.3330865232016633, 1.3337136754864414, 1.3343408457171484, 1.3349680337264078, 1.3355952393468438, 1.3362224624110803, 1.3368497027517416, 1.3374769602014516, 1.3381042345928338, 1.3387315257585126, 1.339358833531112, 1.3399861577432557, 1.3406134982275679, 1.3412408548166723, 1.3418682273431934, 1.3424956156397547, 1.3431230195389803, 1.3437504388734942, 1.3443778734759204, 1.3450053231788828, 1.3456327878150056, 1.3462602672169124, 1.3468877612172276, 1.3475152696485748, 1.348142792343578, 1.3487703291348616, 1.3493978798550492, 1.3500254443367647, 1.3506530224126323, 1.351280613915276, 1.3519082186773197, 1.3525358365313871, 1.3531634673101027, 1.3537911108460901, 1.3544187669719734, 1.3550464355203764, 1.3556741163239234, 1.3563018092152384, 1.3569295140269448, 1.3575572305916672, 1.3581849587420292, 1.358812698310655, 1.3594404491301684, 1.3600682110331934, 1.3606959838523542, 1.3613237674202747, 1.3619515615695785, 1.36257936613289, 1.363207180942833, 1.3638350058320317, 1.3644628406331096, 1.3650906851786913, 1.3657185393014002, 1.3663464028338606, 1.3669742756086962, 1.3676021574585313, 1.3682300482159895, 1.3688579477136953, 1.3694858557842724, 1.3701137722603447, 1.3707416969745363, 1.3713696297594709, 1.3719975704477729, 1.372625518872066, 1.3732534748649741, 1.3738814382591213, 1.3745094088871319, 1.3751373865816294, 1.375765371175238, 1.3763933625005813, 1.3770213603902839, 1.3776493646769694, 1.378277375193262, 1.3789053917717853, 1.3795334142451634, 1.3801614424460205, 1.3807894762069806, 1.3814175153606674, 1.382045559739705, 1.3826736091767171, 1.3833016635043283, 1.3839297225551621, 1.3845577861618426, 1.3851858541569937, 1.3858139263732396, 1.3864420026432038, 1.3870700827995108, 1.3876981666747845, 1.3883262541016486, 1.3889543449127273, 1.3895824389406444, 1.3902105360180239, 1.39083863597749, 1.3914667386516664, 1.3920948438731773, 1.3927229514746466, 1.3933510612886981, 1.3939791731479561, 1.3946072868850443, 1.3952354023325868, 1.3958635193232074, 1.3964916376895304, 1.3971197572641794, 1.3977478778797789, 1.3983759993689522, 1.3990041215643239, 1.3996322442985174, 1.4002603674041572, 1.400888490713867, 1.4015166140602711, 1.4021447372759928, 1.4027728601936564, 1.4034009826458862, 1.4040291044653059, 1.4046572254845395, 1.4052853455362109, 1.4059134644529441, 1.4065415820673635, 1.4071696982120923, 1.407797812719755, 1.4084259254229754, 1.4090540361543775, 1.4096821447465855, 1.410310251032223, 1.4109383548439141, 1.4115664560142831, 1.4121945543759535, 1.4128226497615495, 1.413450742003695, 1.4140788309350141, 1.414706916388131, 1.4153349981956689, 1.4159630761902524, 1.4165911502045054, 1.4172192200710518, 1.4178472856225155, 1.4184753466915208, 1.4191034031106911, 1.4197314547126507, 1.4203595013300239, 1.4209875427954342, 1.4216155789415057, 1.4222436096008624, 1.4228716346061283, 1.4234996537899272, 1.4241276669848835, 1.4247556740236207, 1.4253836747387632, 1.4260116689629347, 1.4266396565287591, 1.4272676372688606, 1.4278956110158632, 1.4285235776023906, 1.4291515368610668, 1.4297794886245161, 1.4304074327253624, 1.4310353689962292, 1.4316632972697412, 1.4322912173785218, 1.432919129155195, 1.4335470324323853, 1.4341749270427164, 1.4348028128188119, 1.4354306895932964, 1.4360585571987934, 1.4366864154679271, 1.4373142642333212, 1.4379421033276001, 1.4385699325833874, 1.4391977518333074, 1.4398255609099837, 1.4404533596460407, 1.4410811478741024, 1.4417089254267919, 1.4423366921367342, 1.4429644478365529, 1.4435921923588719, 1.4442199255363153, 1.4448476472015068, 1.4454753571870709, 1.4461030553256311, 1.4467307414498114, 1.4473584153922361, 1.4479860769855293, 1.4486137260623142, 1.4492413624552154, 1.4498689859968565, 1.450496596519862, 1.4511241938568555, 1.4517517778404609, 1.4523793483033023, 1.4530069050780039, 1.4536344479971892, 1.4542619768934828, 1.4548894915995081, 1.4555169919478894, 1.4561444777712504, 1.4567719489022153, 1.4573994051734083, 1.4580268464174528, 1.4586542724669731, 1.4592816831545932, 1.4599090783129369, 1.4605364577746283, 1.4611638213722915, 1.4617911689385503, 1.4624185003060288, 1.4630458153073507, 1.4636731137751404, 1.4643003955420215, 1.4649276604406181, 1.4655549083035544, 1.466182138963454, 1.4668093522529411, 1.4674365480046396, 1.4680637260511735, 1.4686908862251666, 1.4693180283592433, 1.4699451522860274, 1.4705722578381426, 1.471199344848213, 1.4718264131488628, 1.4724534625727159, 1.4730804929523962, 1.4737075041205276, 1.474334495909734, 1.4749614681526397, 1.4755884206818686, 1.4762153533300442, 1.4768422659297911, 1.4774691583137329, 1.4780960303144939, 1.478722881764698, 1.4793497124969686, 1.4799765223439305, 1.4806033111382071, 1.4812300787124228, 1.481856824899201, 1.4824835495311663, 1.4831102524409423, 1.4837369334611532, 1.484363592424423, 1.4849902291633752, 1.4856168435106341, 1.4862434352988236, 1.486870004360568, 1.4874965505284909, 1.4881230736352165, 1.4887495735133687, 1.4893760499955715, 1.4900025029144488, 1.4906289321026245, 1.4912553373927229, 1.4918817186173674, 1.4925080756091826, 1.4931344082007922, 1.4937607162248201, 1.4943869995138903, 1.495013257900627, 1.4956394912176538, 1.4962656992975951, 1.4968918819730745, 1.4975180390767162, 1.4981441704411442, 1.4987702758989823, 1.4993963552828544, 1.5000224084253848, 1.5006484351591973, 1.5012744353169158, 1.5019004087311645, 1.5025263552345671, 1.5031522746597479, 1.5037781668393306, 1.5044040316059393, 1.5050298687921979, 1.5056556782307302, 1.5062814597541607, 1.5069072131951131, 1.5075329383862111, 1.5081586351600791, 1.5087843033493407, 1.5094099427866201, 1.5100355533045413, 1.5106611347357282, 1.5112866869128045, 1.5119122096683948, 1.512537702835123, 1.5131631662456124, 1.5137885997324876, 1.5144140031283722, 1.5150393762658902])
        else:
            flat = np.ma.masked_array(data=DomeFlat_hk / np.nanmedian(DomeFlat_hk), mask=bad_pixel_mask)
            arc_ar = ArcAr_hk / flat
            arc_xe = ArcXe_hk / flat
            wv = np.asarray([1.4002728036019922, 1.4011826721593568, 1.4020927686081217, 1.4030030926405377, 1.4039136439488551, 1.4048244222253248, 1.4057354271621973, 1.4066466584517232, 1.4075581157861532, 1.4084697988577379, 1.409381707358728, 1.4102938409813741, 1.4112061994179266, 1.4121187823606365, 1.413031589501754, 1.41394462053353, 1.414857875148215, 1.4157713530380598, 1.4166850538953148, 1.4175989774122306, 1.4185131232810582, 1.4194274911940479, 1.4203420808434504, 1.4212568919215161, 1.422171924120496, 1.4230871771326405, 1.4240026506502004, 1.4249183443654259, 1.4258342579705681, 1.4267503911578774, 1.4276667436196047, 1.4285833150480001, 1.4295001051353144, 1.4304171135737986, 1.431334340055703, 1.4322517842732783, 1.4331694459187749, 1.4340873246844437, 1.4350054202625353, 1.4359237323453002, 1.4368422606249891, 1.4377610047938527, 1.4386799645441413, 1.4395991395681058, 1.440518529557997, 1.4414381342060651, 1.4423579532045609, 1.443277986245735, 1.4441982330218381, 1.445118693225121, 1.4460393665478337, 1.4469602526822276, 1.4478813513205526, 1.4488026621550598, 1.4497241848779998, 1.450645919181623, 1.4515678647581802, 1.4524900212999219, 1.4534123884990988, 1.4543349660479614, 1.4552577536387605, 1.4561807509637468, 1.4571039577151705, 1.4580273735852827, 1.4589509982663336, 1.4598748314505741, 1.4607988728302548, 1.4617231220976263, 1.4626475789449391, 1.463572243064444, 1.4644971141483913, 1.4654221918890322, 1.4663474759786168, 1.4672729661093959, 1.4681986619736203, 1.4691245632635401, 1.4700506696714066, 1.4709769808894699, 1.4719034966099809, 1.4728302165251901, 1.4737571403273482, 1.4746842677087058, 1.4756115983615135, 1.4765391319780219, 1.4774668682504815, 1.4783948068711432, 1.4793229475322576, 1.4802512899260751, 1.4811798337448463, 1.4821085786808221, 1.483037524426253, 1.4839666706733896, 1.4848960171144825, 1.4858255634417823, 1.4867553093475396, 1.4876852545240051, 1.4886153986634296, 1.4895457414580635, 1.4904762826001572, 1.4914070217819619, 1.4923379586957277, 1.4932690930337054, 1.4942004244881457, 1.4951319527512992, 1.4960636775154164, 1.4969955984727481, 1.4979277153155448, 1.4988600277360571, 1.4997925354265356, 1.5007252380792311, 1.5016581353863943, 1.5025912270402753, 1.5035245127331252, 1.5044579921571946, 1.5053916650047339, 1.5063255309679937, 1.5072595897392249, 1.5081938410106779, 1.5091282844746035, 1.510062919823252, 1.5109977467488744, 1.5119327649437213, 1.5128679741000428, 1.5138033739100902, 1.5147389640661137, 1.5156747442603642, 1.516610714185092, 1.5175468735325479, 1.5184832219949826, 1.5194197592646466, 1.5203564850337905, 1.5212933989946651, 1.5222305008395207, 1.5231677902606082, 1.5241052669501782, 1.5250429306004814, 1.525980780903768, 1.5269188175522892, 1.5278570402382949, 1.5287954486540365, 1.5297340424917643, 1.5306728214437288, 1.5316117852021807, 1.5325509334593705, 1.5334902659075493, 1.5344297822389672, 1.535369482145875, 1.5363093653205233, 1.5372494314551628, 1.5381896802420441, 1.5391301113734177, 1.5400707245415344, 1.5410115194386447, 1.5419524957569992, 1.5428936531888486, 1.5438349914264435, 1.5447765101620345, 1.5457182090878725, 1.5466600878962078, 1.5476021462792908, 1.5485443839293727, 1.5494868005387037, 1.5504293957995345, 1.5513721694041158, 1.5523151210446984, 1.5532582504135326, 1.5542015572028691, 1.5551450411049585, 1.5560887018120517, 1.557032539016399, 1.5579765524102511, 1.5589207416858588, 1.5598651065354725, 1.5608096466513428, 1.5617543617257206, 1.5626992514508562, 1.5636443155190003, 1.5645895536224037, 1.5655349654533168, 1.5664805507039905, 1.5674263090666751, 1.5683722402336215, 1.5693183438970801, 1.5702646197493018, 1.5712110674825368, 1.5721576867890361, 1.5731044773610503, 1.5740514388908298, 1.5749985710706251, 1.5759458735926872, 1.5768933461492667, 1.5778409884326141, 1.5787888001349799, 1.5797367809486147, 1.5806849305657695, 1.5816332486786946, 1.5825817349796407, 1.5835303891608585, 1.5844792109145984, 1.5854281999331112, 1.5863773559086476, 1.5873266785334581, 1.5882761674997932, 1.5892258224999039, 1.5901756432260403, 1.5911256293704534, 1.5920757806253938, 1.593026096683112, 1.5939765772358587, 1.5949272219758843, 1.5958780305954399, 1.5968290027867758, 1.5977801382421424, 1.5987314366537908, 1.5996828977139712, 1.6006345211149346, 1.6015863065489313, 1.6025382537082122, 1.6034903622850278, 1.6044426319716287, 1.6053950624602653, 1.6063476534431889, 1.6073004046126493, 1.6082533156608976, 1.6092063862801842, 1.61015961616276, 1.6111130050008755, 1.6120665524867812, 1.6130202583127278, 1.613974122170966, 1.6149281437537462, 1.6158823227533192, 1.6168366588619358, 1.6177911517718462, 1.6187458011753013, 1.6197006067645516, 1.620655568231848, 1.6216106852694407, 1.6225659575695806, 1.6235213848245182, 1.6244769667265042, 1.6254327029677893, 1.6263885932406239, 1.6273446372372586, 1.6283008346499444, 1.6292571851709314, 1.6302136884924707, 1.6311703443068128, 1.6321271523062082, 1.6330841121829074, 1.6340412236291613, 1.6349984863372204, 1.6359558999993355, 1.6369134643077567, 1.6378711789547353, 1.6388290436325215, 1.639787058033366, 1.6407452218495195, 1.6417035347732325, 1.6426619964967557, 1.6436206067123396, 1.6445793651122351, 1.6455382713886926, 1.6464973252339627, 1.6474565263402963, 1.6484158743999435, 1.6493753691051554, 1.6503350101481824, 1.6512947972212753, 1.6522547300166845, 1.6532148082266609, 1.6541750315434549, 1.6551353996593172, 1.6560959122664982, 1.6570565690572487, 1.6580173697238196, 1.6589783139584611, 1.6599394014534241, 1.6609006319009589, 1.6618620049933166, 1.6628235204227473, 1.6637851778815018, 1.6647469770618311, 1.6657089176559854, 1.6666709993562154, 1.6676332218547718, 1.6685955848439051, 1.669558088015866, 1.6705207310629051, 1.671483513677273, 1.6724464355512205, 1.6734094963769981, 1.6743726958468563, 1.6753360336530461, 1.6762995094878175, 1.6772631230434216, 1.6782268740121089, 1.6791907620861302, 1.6801547869577358, 1.6811189483191764, 1.6820832458627026, 1.6830476792805653, 1.6840122482650151, 1.6849769525083023, 1.6859417917026773, 1.6869067655403915, 1.6878718737136951, 1.6888371159148385, 1.6898024918360728, 1.6907680011696484, 1.6917336436078159, 1.6926994188428257, 1.6936653265669288, 1.6946313664723758, 1.695597538251417, 1.6965638415963031, 1.6975302761992852, 1.6984968417526134, 1.6994635379485383, 1.7004303644793111, 1.7013973210371818, 1.7023644073144011, 1.7033316230032198, 1.7042989677958886, 1.7052664413846581, 1.7062340434617789, 1.7072017737195013, 1.7081696318500763, 1.7091376175457542, 1.7101057304987861, 1.7110739704014222, 1.7120423369459135, 1.7130108298245101, 1.7139794487294631, 1.7149481933530228, 1.7159170633874399, 1.7168860585249652, 1.7178551784578493, 1.7188244228783425, 1.7197937914786958, 1.7207632839511597, 1.7217328999879846, 1.7227026392814215, 1.7236725015237206, 1.724642486407133, 1.7256125936239088, 1.7265828228662992, 1.7275531738265544, 1.7285236461969251, 1.729494239669662, 1.7304649539370156, 1.7314357886912366, 1.7324067436245758, 1.7333778184292834, 1.7343490127976104, 1.7353203264218076, 1.7362917589941249, 1.7372633102068136, 1.7382349797521237, 1.7392067673223064, 1.7401786726096122, 1.7411506953062914, 1.7421228351045952, 1.7430950916967736, 1.7440674647750776, 1.7450399540317576, 1.7460125591590643, 1.7469852798492487, 1.7479581157945607, 1.7489310666872515, 1.7499041322195714, 1.7508773120837713, 1.7518506059721015, 1.7528240135768129, 1.7537975345901557, 1.7547711687043812, 1.7557449156117397, 1.7567187750044815, 1.7576927465748577, 1.7586668300151187, 1.759641025017515, 1.7606153312742976, 1.7615897484777165, 1.7625642763200231, 1.7635389144934674, 1.7645136626903004, 1.7654885206027726, 1.7664634879231345, 1.7674385643436368, 1.7684137495565302, 1.7693890432540651, 1.7703644451284926, 1.7713399548720627, 1.7723155721770265, 1.7732912967356345, 1.774267128240137, 1.7752430663827852, 1.7762191108558292, 1.7771952613515198, 1.7781715175621078, 1.7791478791798436, 1.7801243458969782, 1.7811009174057615, 1.7820775933984447, 1.7830543735672784, 1.784031257604513, 1.7850082452023992, 1.7859853360531877, 1.786962529849129, 1.7879398262824737, 1.7889172250454726, 1.7898947258303763, 1.7908723283294352, 1.7918500322349002, 1.7928278372390216, 1.7938057430340504, 1.7947837493122369, 1.7957618557658319, 1.7967400620870859, 1.7977183679682498, 1.7986967731015739, 1.799675277179309, 1.8006538798937055, 1.8016325809370144, 1.802611380001486, 1.803590276779371, 1.80456927096292, 1.8055483622443838, 1.806527550316013, 1.807506834870058, 1.8084862155987693, 1.8094656921943981, 1.8104452643491946, 1.8114249317554094, 1.8124046941052934, 1.8133845510910969, 1.8143645024050707, 1.8153445477394654, 1.8163246867865317, 1.81730491923852, 1.8182852447876812, 1.8192656631262656, 1.8202461739465241, 1.8212267769407073, 1.8222074718010657, 1.82318825821985, 1.8241691358893106, 1.8251501045016987, 1.8261311637492641, 1.8271123133242582, 1.8280935529189311, 1.8290748822255336, 1.8300563009363162, 1.8310378087435299, 1.832019405339425, 1.8330010904162521, 1.8339828636662621, 1.8349647247817051, 1.8359466734548324, 1.8369287093778941, 1.837910832243141, 1.8388930417428238, 1.839875337569193, 1.8408577194144993, 1.8418401869709933, 1.8428227399309254, 1.8438053779865466, 1.8447881008301072, 1.8457709081538582, 1.84675379965005, 1.847736775010933, 1.8487198339287583, 1.8497029760957762, 1.8506862012042373, 1.8516695089463924, 1.8526528990144917, 1.8536363711007864, 1.854619924897527, 1.8556035600969638, 1.8565872763913478, 1.8575710734729292, 1.858554951033959, 1.8595389087666878, 1.8605229463633659, 1.8615070635162441, 1.8624912599175731, 1.8634755352596035, 1.8644598892345861, 1.8654443215347709, 1.8664288318524092, 1.8674134198797514, 1.8683980853090478, 1.8693828278325495, 1.870367647142507, 1.8713525429311706, 1.8723375148907915, 1.8733225627136196, 1.8743076860919063, 1.8752928847179016, 1.8762781582838564, 1.8772635064820213, 1.8782489290046469, 1.8792344255439839, 1.8802199957922827, 1.881205639441794, 1.8821913561847685, 1.883177145713457, 1.8841630077201099, 1.8851489418969778, 1.8861349479363114, 1.8871210255303612, 1.888107174371378, 1.8890933941516121, 1.8900796845633148, 1.891066045298736, 1.8920524760501267, 1.8930389765097373, 1.8940255463698188, 1.8950121853226214, 1.8959988930603959, 1.8969856692753928, 1.8979725136598631, 1.8989594259060569, 1.8999464057062252, 1.9009334527526187, 1.9019205667374877, 1.9029077473530827, 1.9038949942916548, 1.9048823072454542, 1.9058696859067319, 1.9068571299677384, 1.907844639120724, 1.9088322130579398, 1.909819851471636, 1.9108075540540637, 1.9117953204974731, 1.9127831504941151, 1.9137710437362399, 1.9147589999160985, 1.9157470187259416, 1.9167350998580195, 1.9177232430045832, 1.9187114478578828, 1.9196997141101693, 1.9206880414536933, 1.9216764295807054, 1.9226648781834561, 1.9236533869541961, 1.9246419555851761, 1.9256305837686467, 1.9266192711968584, 1.9276080175620618, 1.9285968225565078, 1.9295856858724467, 1.9305746072021295, 1.9315635862378064, 1.9325526226717282, 1.9335417161961455, 1.934530866503309, 1.9355200732854692, 1.9365093362348769, 1.9374986550437825, 1.9384880294044369, 1.9394774590090904, 1.9404669435499939, 1.9414564827193979, 1.9424460762095528, 1.9434357237127098, 1.944425424921119, 1.9454151795270311, 1.9464049872226969, 1.9473948477003669, 1.9483847606522917, 1.9493747257707223, 1.9503647427479087, 1.9513548112761019, 1.9523449310475525, 1.9533351017545111, 1.9543253230892281, 1.9553155947439542, 1.9563059164109404, 1.9572962877824371, 1.9582867085506948, 1.9592771784079641, 1.9602676970464958, 1.9612582641585405, 1.9622488794363486, 1.9632395425721709, 1.9642302532582583, 1.9652210111868609, 1.9662118160502295, 1.9672026675406149, 1.9681935653502676, 1.9691845091714382, 1.9701754986963773, 1.9711665336173354, 1.9721576136265635, 1.9731487384163122, 1.9741399076788315, 1.9751311211063727, 1.9761223783911861, 1.9771136792255224, 1.9781050233016322, 1.9790964103117661, 1.9800878399481747, 1.9810793119031089, 1.9820708258688191, 1.9830623815375557, 1.9840539786015698, 1.9850456167531116, 1.9860372956844321, 1.9870290150877814, 1.9880207746554106, 1.9890125740795701, 1.9900044130525107, 1.9909962912664829, 1.991988208413737, 1.9929801641865244, 1.9939721582770948, 1.9949641903776998, 1.9959562601805891, 1.996948367378014, 1.9979405116622249, 1.998932692725472, 1.9999249102600065, 2.0009171639580789, 2.0019094535119395, 2.0029017786138397, 2.0038941389560287, 2.0048865342307591, 2.0058789641302797, 2.0068714283468418, 2.0078639265726963, 2.0088564585000936, 2.0098490238212845, 2.0108416222285195, 2.0118342534140488, 2.0128269170701234, 2.013819612888994, 2.0148123405629113, 2.0158050997841253, 2.0167978902448875, 2.0177907116374478, 2.0187835636540572, 2.0197764459869663, 2.0207693583284261, 2.0217623003706859, 2.0227552718059978, 2.0237482723266118, 2.0247413016247782, 2.0257343593927484, 2.0267274453227722, 2.027720559107101, 2.0287137004379847, 2.0297068690076743, 2.0307000645084203, 2.0316932866324739, 2.0326865350720849, 2.0336798095195041, 2.0346731096669823, 2.0356664352067702, 2.0366597858311186, 2.0376531612322775, 2.0386465611024978, 2.0396399851340301, 2.0406334330191251, 2.0416269044500339, 2.0426203991190062, 2.043613916718293, 2.0446074569401453, 2.0456010194768135, 2.046594604020548, 2.0475882102635992, 2.0485818378982183, 2.0495754866166558, 2.0505691561111621, 2.0515628460739883, 2.0525565561973842, 2.0535502861736008, 2.0545440356948892, 2.0555378044534995, 2.0565315921416825, 2.0575253984516886, 2.058519223075769, 2.0595130657061733, 2.0605069260351532, 2.0615008037549587, 2.0624946985578405, 2.0634886101360497, 2.0644825381818359, 2.0654764823874507, 2.0664704424451443, 2.0674644180471677, 2.0684584088857707, 2.0694524146532047, 2.07044643504172, 2.0714404697435675, 2.0724345184509971, 2.0734285808562603, 2.0744226566516071, 2.0754167455292887, 2.0764108471815552, 2.0774049613006573, 2.0783990875788461, 2.0793932257083716, 2.080387375381485, 2.081381536290436, 2.0823757081274761, 2.0833698905848559, 2.0843640833548251, 2.0853582861296358, 2.0863524986015376, 2.0873467204627811, 2.088340951405617, 2.0893351911222964, 2.0903294393050693, 2.0913236956461869, 2.0923179598378994, 2.0933122315724577, 2.0943065105421121, 2.0953007964391137, 2.0962950889557126, 2.0972893877841594, 2.0982836926167052, 2.0992780031456006, 2.1002723190630959, 2.1012666400614419, 2.1022609658328886, 2.1032552960696878, 2.1042496304640892, 2.105243968708344, 2.1062383104947022, 2.1072326555154151, 2.1082270034627331, 2.1092213540289064, 2.1102157069061862, 2.1112100617868226, 2.1122044183630662, 2.1131987763271685, 2.1141931353713792, 2.1151874951879495, 2.1161818554691294, 2.1171762159071701, 2.1181705761943221, 2.1191649360228357, 2.1201592950849619, 2.1211536530729513, 2.1221480096790541, 2.1231423645955214, 2.1241367175146038, 2.1251310681285513, 2.1261254161296153, 2.1271197612100461, 2.1281141030620945, 2.1291084413780106, 2.1301027758500455, 2.1310971061704498, 2.1320914320314737, 2.1330857531253686, 2.1340800691443844, 2.1350743797807721, 2.1360686847267818, 2.1370629836746651, 2.1380572763166716, 2.1390515623450526, 2.1400458414520589, 2.1410401133299404, 2.1420343776709476, 2.1430286341673321, 2.1440228825113437, 2.1450171223952332, 2.1460113535112515, 2.1470055755516491, 2.1479997882086765, 2.1489939911745846, 2.1499881841416233, 2.1509823668020442, 2.1519765388480971, 2.1529706999720335, 2.153964849866103, 2.1549589882225568, 2.1559531147336459, 2.1569472290916201, 2.1579413309887303, 2.1589354201172273, 2.1599294961693616, 2.1609235588373839, 2.1619176078135451, 2.1629116427900952, 2.163905663459285, 2.1648996695133653, 2.1658936606445867, 2.1668876365451997, 2.1678815969074554, 2.1688755414236036, 2.1698694697858953, 2.1708633816865817, 2.1718572768179123, 2.1728511548721388, 2.1738450155415112, 2.1748388585182798, 2.1758326834946962, 2.1768264901630103, 2.1778202782154734, 2.1788140473443351, 2.1798077972418466, 2.1808015276002588, 2.1817952381118215, 2.1827889284687862, 2.1837825983634032, 2.1847762474879229, 2.1857698755345965, 2.1867634821956736, 2.1877570671634059, 2.1887506301300435, 2.1897441707878373, 2.1907376888290373, 2.1917311839458948, 2.1927246558306601, 2.1937181041755838, 2.1947115286729164, 2.1957049290149091, 2.196698304893812, 2.1976916560018758, 2.1986849820313514, 2.1996782826744892, 2.2006715576235396, 2.2016648065707534, 2.2026580292083815, 2.2036512252286742, 2.204644394323882, 2.2056375361862566, 2.2066306505080471, 2.2076237369815046, 2.2086167952988802, 2.2096098251524241, 2.2106028262343873, 2.21159579823702, 2.2125887408525728, 2.2135816537732969, 2.2145745366914422, 2.21556738929926, 2.2165602112890004, 2.217553002352914, 2.2185457621832518, 2.2195384904722646, 2.2205311869122024, 2.2215238511953159, 2.2225164830138562, 2.2235090820600734, 2.2245016480262185, 2.2254941806045418, 2.2264866794872944, 2.2274791443667263, 2.2284715749350887, 2.2294639708846322, 2.2304563319076065, 2.2314486576962635, 2.2324409479428531, 2.2334332023396257, 2.234425420578833, 2.2354176023527241, 2.2364097473535507, 2.2374018552735633, 2.2383939258050125, 2.2393859586401486, 2.2403779534712225, 2.2413699099904845, 2.2423618278901856, 2.2433537068625764, 2.2443455465999071, 2.2453373467944289, 2.2463291071383917, 2.2473208273240473, 2.2483125070436452, 2.2493041459894361, 2.2502957438536715, 2.2512873003286011, 2.2522788151064761, 2.2532702878795465, 2.2542617183400635, 2.2552531061802776, 2.2562444510924395, 2.2572357527687998, 2.2582270109016083, 2.2592182251831172, 2.2602093953055755, 2.2612005209612351, 2.2621916018423458, 2.2631826376411586, 2.2641736280499241, 2.2651645727608929, 2.2661554714663152, 2.2671463238584426, 2.2681371296295247, 2.2691278884718127, 2.2701186000775571, 2.2711092641390085, 2.2720998803484171, 2.2730904483980345, 2.2740809679801104, 2.2750714387868962, 2.2760618605106417, 2.277052232843598, 2.2780425554780157, 2.2790328281061454, 2.2800230504202377, 2.2810132221125432, 2.2820033428753121, 2.282993412400796, 2.283983430381245, 2.2849733965089092, 2.2859633104760402, 2.2869531719748881, 2.2879429806977032, 2.2889327363367369, 2.2899224385842389, 2.2909120871324609, 2.2919016816736528, 2.292891221900065, 2.2938807075039489, 2.2948701381775547, 2.295859513613133, 2.2968488335029345, 2.2978380975392096, 2.2988273054142092, 2.2998164568201838, 2.3008055514493839, 2.3017945889940608, 2.302783569146464, 2.3037724915988447, 2.3047613560434539, 2.3057501621725418, 2.306738909678359, 2.3077275982531562, 2.3087162275891844, 2.3097047973786933, 2.3106933073139344, 2.3116817570871575, 2.3126701463906141, 2.3136584749165547, 2.3146467423572292, 2.315634948404889, 2.3166230927517844, 2.3176111750901658, 2.318599195112284, 2.3195871525103899, 2.3205750469767334, 2.3215628782035664, 2.3225506458831382, 2.3235383497077002, 2.3245259893695023, 2.3255135645607963, 2.3265010749738315, 2.3274885203008595, 2.3284759002341304, 2.3294632144658953, 2.3304504626884039, 2.3314376445939082, 2.3324247598746575, 2.3334118082229032, 2.3343987893308955, 2.3353857028908855, 2.3363725485951234, 2.3373593261358598, 2.3383460352053458, 2.3393326754958315, 2.3403192466995675, 2.3413057485088054, 2.3422921806157944, 2.3432785427127856, 2.3442648344920305, 2.3452510556457788, 2.3462372058662808, 2.3472232848457883, 2.3482092922765512, 2.3491952278508204, 2.3501810912608461, 2.3511668821988794, 2.3521526003571704, 2.3531382454279699, 2.3541238171035292, 2.3551093150760978, 2.356094739037927, 2.3570800886812675, 2.3580653636983695, 2.359050563781484, 2.3600356886228617, 2.3610207379147523, 2.3620057113494077, 2.3629906086190777, 2.3639754294160134, 2.3649601734324648, 2.3659448403606831, 2.3669294298929189, 2.3679139417214223, 2.3688983755384445, 2.3698827310362356, 2.3708670079070466, 2.3718512058431287, 2.3728353245367311, 2.3738193636801053, 2.3748033229655019, 2.3757872020851711, 2.3767710007313645, 2.3777547185963313, 2.3787383553723234, 2.3797219107515906, 2.3807053844263844, 2.3816887760889545, 2.3826720854315515, 2.383655312146427, 2.3846384559258302, 2.3856215164620131, 2.3866044934472259, 2.3875873865737187, 2.3885701955337431, 2.3895529200195487, 2.3905355597233866, 2.3915181143375071, 2.3925005835541615, 2.3934829670655997, 2.3944652645640732, 2.3954474757418316])

        wavs[gr] = wv

        # add arcs to all_data dict for sum function
        all_data['ArcAr'+gr] = {'flat_cor_imgs':[arc_ar]}
        all_data['ArcXe'+gr] = {'flat_cor_imgs':[arc_xe]}

        for oname in ['comp', 'targ']:      # each star
            exp_time = float(data[gr][oname].split('_')[-1])
            for nod in ['A', 'B']:          # each nod point
                obj = nod + data[gr][oname]     # object name
                s_row = get_mean_row(obj)       # mean row of spectrum

                nimgs = len(all_data[obj]['raw_imgs'])
                for i in range(nimgs):          # flat field correct each image
                    fimg = all_data[obj]['raw_imgs'][i] / flat

                    if nod == 'A':
                        o_nod = 'B'
                    else:
                        o_nod = 'A'
                    # fimg -= (all_data[obj.replace(nod, o_nod)]['raw_imgs'][i] / flat)   # sky subtraction by nod points

                    fimg = remove_sky(fimg, s_row, mask=bad_pixel_mask)   # sky subtraction by aperture

                    all_data[obj]['flat_cor_imgs'].append(fimg)
                    # plot_im(fimg)

                print "  >", obj, "| mean row = %d." % s_row

                # arc_row_ar = sum_spectra('ArcAr'+gr, s_row, imgs='flat_cor_imgs')     # arc spectra
                # arc_row_xe = sum_spectra('ArcXe'+gr, s_row, imgs='flat_cor_imgs')

                # wv = get_wav_sol(arc_row_ar, arc_row_xe, grism=gr, tol=0.02
                #                  # , plot_resid=True, plot_arcs=False, plot_fit=True
                #                  )    # wavelength solution for object

                spc = np.asarray(sum_spectra(obj, s_row, method='optimal', imgs='flat_cor_imgs', mask=bad_pixel_mask))  # object spectrum

                # plt.plot(wv, spc)
                # plt.show()

                # spc = normalise(spc)
                specs[gr][oname][nod] = spc / exp_time
                # plt.plot(wv, spc, label=obj, alpha=0.8)

            coadd_spc = ((specs[gr][oname]['A'] + specs[gr][oname]['B']) / 2.0)      # combine nod points
            specs[gr][oname]['both'] = coadd_spc
            # plt.plot(wv, normalise(coadd_spc), label=oname+'_'+gr)

    with open('extract.pkl', 'wb') as pf:
        pickle.dump([specs, wavs], pf)

else:
    print "> Loading spectra from past extraction ..."
    with open('extract.pkl', 'rb') as pf:
        specs, wavs = pickle.load(pf)


wav_grid = {'hk':np.linspace(min(wavs['hk']), max(wavs['hk']), len(wavs['hk'])),
            'zj':np.linspace(min(wavs['zj']), max(wavs['zj']), len(wavs['zj']))}

for gr in ['hk', 'zj']:
    for star in ['comp', 'targ']:
        for nod in ['A', 'B', 'both']:
            specs[gr][star][nod] = rebin_spec(wavs[gr], specs[gr][star][nod], wav_grid[gr])
    wavs[gr] = wav_grid[gr]

m_w, m_s = load_model_spec('data/9400_5.0.fits', [0.8, 2.5])
# m_f = np.genfromtxt('data/uka0v.dat').T
# m_w, m_s = m_f[0]/1e4, m_f[1]

trends = {}
plt.figure(figsize=(10, 7))
for gr in ['hk', 'zj']:
    # plt.plot(wavs[gr], specs[gr]['comp']['both'], alpha=0.8, label='HD27267_'+gr)

    for nod in ['A', 'B']:
        plt.plot(wavs[gr], specs[gr]['comp'][nod], alpha=0.8, label='HD27267_'+gr+'_'+nod, color='grey')

    m_s_r = rebin_spec(m_w, m_s, wavs[gr], convolve=True) / 5e11     #* 1e4    #* 5e3 #/ 1e12
    plt.plot(wavs[gr], m_s_r, color='k', label='A0 model')

    plot_transmission(factor=2000.0)
    # plot_throughput(factor=2000.0)

    trends[gr] = specs[gr]['comp']['both'] / m_s_r
# plt.legend()
# plt.show()
plt.savefig('comp.png', format='png')
plt.clf()

plt.figure(figsize=(10, 7))
for teff in ['3400', '3600', '3800', '4000']:
    m_w, m_s = load_model_spec('data/'+teff+'_5.0.fits', [0.8, 2.5])

    for gr in ['hk', 'zj']:
        # plt.plot(wavs[gr], specs[gr]['targ']['both'] / trends[gr], alpha=0.8, label='LP358_'+gr)
        # plt.plot(wavs[gr], specs[gr]['targ']['both'], alpha=0.8, label='LP358_'+gr+' uncor')
    
        for nod in ['A', 'B']:
            plt.plot(wavs[gr], specs[gr]['targ'][nod] / trends[gr], alpha=0.8, label='LP358_'+gr+'_'+nod, color='grey')
    
        m_s_r = rebin_spec(m_w, m_s, wavs[gr], convolve=True) / 5e11    #1e12
        plt.plot(wavs[gr], m_s_r, color='k', label='Model')

    plot_transmission()
    plt.title(teff)
    plt.legend()
    plt.savefig(teff+'.png', format='png')
    plt.ylim(0, 300)
    plt.clf()
    # plt.show()


# ---------------------------------------------------------------------------------------------------------------------

# wav = wavs[0]   # wavelength of comparison as base
# plt.plot(wav, normalise(specs[0]), label='HD 27267')   # comparison star
# plt.plot(wavs[1], normalise(specs[1]), label='LP uncor')

# stellar model spectra
# a0v_w, a0v_s = load_model_spec('data/9400_5.0.fits', wav_ranges[gr])     # A0V model
# a0v_s_bin = normalise(rebin_spec(a0v_w, a0v_s, wav))       # bin to match base wavelength grid

# trend = normalise(specs[0]) - a0v_s_bin # correct for Earth's atmosphere
# plt.plot(wav, trend, label='Trend-phoenix', color='r')  # trend

# lp_spec = normalise(rebin_spec(wavs[1], specs[1], wav)) - trend
# plt.plot(wav, lp_spec, color='k', label='LP 358-499')   # Target

# plt.plot(wav, a0v_s_bin, label='A0V model', color=cols[0], alpha=0.7)

# esa_a0_w, esa_a0_s = np.asarray([0.88924291866864891, 0.88981189059590982, 0.89038100315572555, 0.89095025618071999, 0.89151964950351736, 0.89208918295674133, 0.89265885637301612, 0.89322866958496561, 0.8937986224252138, 0.89436871472638468, 0.89493894632110216, 0.89550931704199022, 0.89607982672167286, 0.89665047519277408, 0.89722126228791776, 0.89779218783972803, 0.89836325168082876, 0.89893445364384394, 0.89950579356139748, 0.90007727126611359, 0.90064888659061593, 0.90122063936752872, 0.90179252942947585, 0.90236455660908121, 0.90293672073896891, 0.90350902165176294, 0.90408145918008709, 0.90465403315656556, 0.90522674341382214, 0.90579958978448083, 0.90637257210116573, 0.90694569019650073, 0.90751894390310983, 0.90809233305361692, 0.9086658574806461, 0.90923951701682137, 0.90981331149476652, 0.91038724074710564, 0.91096130460646274, 0.9115355029054617, 0.91210983547672664, 0.91268430215288143, 0.91325890276654997, 0.91383363715035637, 0.91440850513692451, 0.9149835065588785, 0.91555864124884223, 0.91613390903943959, 0.91670930976329468, 0.9172848432530315, 0.91786050934127394, 0.9184363078606459, 0.91901223864377157, 0.91958830152327475, 0.92016449633177944, 0.92074082290190962, 0.92131728106628941, 0.92189387065754258, 0.92247059150829325, 0.92304744345116529, 0.92362442631878272, 0.92420153994376952, 0.9247787841587497, 0.92535615879634725, 0.92593366368918595, 0.92651129866989002, 0.92708906357108334, 0.9276669582253898, 0.92824498246543352, 0.92882313612383838, 0.92940141903322837, 0.9299798310262275, 0.93055837193545976, 0.93113704159354915, 0.93171583983311945, 0.93229476648679477, 0.93287382138719921, 0.93345300436695655, 0.93403231525869079, 0.93461175389502604, 0.93519132010858619, 0.93577101373199512, 0.93635083459787694, 0.93693078253885564, 0.93751085738755502, 0.93809105897659928, 0.93867138713861231, 0.939251841706218, 0.93983242251204036, 0.94041312938870347, 0.94099396216883113, 0.94157492068504756, 0.94215600476997641, 0.94273721425624202, 0.94331854897646805, 0.94390000876327862, 0.94448159344929772, 0.94506330286714935, 0.94564513684945739, 0.94622709522884585, 0.94680917783793872, 0.94739138450935989, 0.94797371507573358, 0.94855616936968346, 0.94913874722383362, 0.94972144847080819, 0.95030427294323094, 0.95088722047372587, 0.95147029089491708, 0.95205348403942847, 0.95263679973988402, 0.95322023782890763, 0.95380379813912342, 0.95438748050315525, 0.95497128475362714, 0.95555521072316307, 0.95613925824438695, 0.95672342714992287, 0.95730771727239472, 0.9578921284444265, 0.95847666049864222, 0.95906131326766586, 0.95964608658412132, 0.96023098028063258, 0.96081599418982366, 0.96140112814431855, 0.96198638197674113, 0.96257175551971552, 0.9631572486058656, 0.96374286106781537, 0.96432859273818872, 0.96491444344960975, 0.96550041303470246, 0.96608650132609064, 0.9666727081563985, 0.96725903335824981, 0.96784547676426869, 0.96843203820707902, 0.9690187175193048, 0.96960551453357002, 0.9701924290824987, 0.9707794609987147, 0.97136661011484204, 0.97195387626350482, 0.97254125927732682, 0.97312875898893214, 0.97371637523094479, 0.97430410783598864, 0.9748919566366876, 0.97547992146566587, 0.97606800215554723, 0.9766561985389558, 0.97724451044851546, 0.9778329377168502, 0.97842148017658404, 0.97901013766034084, 0.97959891000074473, 0.98018779703041958, 0.98077679858198941, 0.98136591448807808, 0.98195514458130972, 0.98254448869430833, 0.98313394665969778, 0.98372351831010207, 0.9843132034781451, 0.98490300199645098, 0.98549291369764369, 0.98608293841434702, 0.98667307597918519, 0.98726332622478197, 0.98785368898376147, 0.98844416408874758, 0.9890347513723643, 0.98962545066723562, 0.99021626180598554, 0.99080718462123796, 0.99139821894561686, 0.99198936461174636, 0.99258062145225023, 0.9931719892997527, 0.99376346798687742, 0.99435505734624863, 0.9949467572104902, 0.99553856741222613, 0.99613048778408031, 0.99672251815867685, 0.99731465836863964, 0.99790690824659267, 0.99849926762515995, 0.99909173633696546, 0.99968431421463311, 1.0002770010907869, 1.0008697967980509, 1.0014627011690487, 1.0020557140364048, 1.002648835232743, 1.0032420645906872, 1.0038354019428612, 1.0044288471218894, 1.0050223999603956, 1.0056160602910036, 1.0062098279463374, 1.0068037027590213, 1.0073976845616788, 1.0079917731869343, 1.0085859684674117, 1.0091802702357346, 1.0097746783245274, 1.010369192566414, 1.0109638127940181, 1.0115585388399639, 1.0121533705368755, 1.0127483077173767, 1.0133433502140916, 1.013938497859644, 1.0145337504866578, 1.0151291079277573, 1.0157245700155662, 1.0163201365827086, 1.0169158074618085, 1.0175115824854899, 1.0181074614863765, 1.0187034442970928, 1.0192995307502621, 1.0198957206785089, 1.0204920139144571, 1.0210884102907305, 1.021684909639953, 1.0222815117947488, 1.0228782165877419, 1.0234750238515562, 1.0240719334188155, 1.0246689451221442, 1.0252660587941658, 1.0258632742675045, 1.0264605913747844, 1.0270580099486293, 1.027655529821663, 1.0282531508265098, 1.0288508727957937, 1.0294486955621382, 1.0300466189581678, 1.0306446428165064, 1.0312427669697777, 1.0318409912506059, 1.0324393154916149, 1.0330377395254289, 1.0336362631846714, 1.0342348863019666, 1.0348336087099388, 1.0354324302412115, 1.0360313507284087, 1.0366303700041548, 1.0372294879010733, 1.0378287042517886, 1.0384280188889243, 1.0390274316451047, 1.0396269423529534, 1.0402265508450947, 1.0408262569541524, 1.0414260605127508, 1.0420259613535134, 1.0426259593090643, 1.0432260542120277, 1.0438262458950274, 1.0444265341906875, 1.0450269189316317, 1.0456273999504844, 1.0462279770798693, 1.0468286501524102, 1.0474294190007316, 1.0480302834574569, 1.0486312433552105, 1.049232298526616, 1.0498334488042977, 1.0504346940208797, 1.0510360340089855, 1.0516374686012395, 1.0522389976302653, 1.0528406209286871, 1.0534423383291289, 1.0540441496642146, 1.0546460547665684, 1.0552480534688138, 1.0558501456035749, 1.0564523310034761, 1.0570546095011411, 1.0576569809291938, 1.0582594451202583, 1.0588620019069586, 1.0594646511219183, 1.0600673925977619, 1.060670226167113, 1.0612731516625957, 1.0618761689168343, 1.0624792777624523, 1.0630824780320738, 1.0636857695583228, 1.0642891521738234, 1.0648926257111992, 1.0654961900030748, 1.0660998448820738, 1.0667035901808199, 1.0673074257319377, 1.0679113513680507, 1.0685153669217831, 1.0691194722257586, 1.0697236671126016, 1.0703279514149358, 1.0709323249653853, 1.0715367875965738, 1.0721413391411254, 1.0727459794316645, 1.0733507083008145, 1.0739555255811997, 1.0745604311054437, 1.075165424706171, 1.0757705062160055, 1.0763756754675706, 1.0769809322934909, 1.0775862765263902, 1.0781917079988923, 1.0787972265436214, 1.0794028319932014, 1.0800085241802562, 1.0806143029374098, 1.0812201680972864, 1.0818261194925096, 1.0824321569557034, 1.0830382803194922, 1.0836444894164996, 1.0842507840793496, 1.0848571641406664, 1.0854636294330737, 1.0860701797891958, 1.0866768150416564, 1.0872835350230794, 1.087890339566089, 1.0884972285033092, 1.089104201667364, 1.0897112588908771, 1.0903184000064725, 1.0909256248467745, 1.0915329332444068, 1.0921403250319934, 1.0927478000421584, 1.0933553581075257, 1.0939629990607194, 1.0945707227343633, 1.0951785289610814, 1.0957864175734977, 1.0963943884042362, 1.0970024412859209, 1.0976105760511756, 1.0982187925326246, 1.0988270905628916, 1.0994354699746007, 1.1000439306003758, 1.1006524722728408, 1.10126109482462, 1.101869798088337, 1.1024785818966161, 1.103087446082081, 1.1036963904773558, 1.1043054149150644, 1.1049145192278309, 1.1055237032482792, 1.1061329668090332, 1.1067423097427171, 1.1073517318819548, 1.10796123305937, 1.1085708131075871, 1.1091804718592297, 1.1097902091469221, 1.1104000248032879, 1.1110099186609514, 1.1116198905525365, 1.1122299403106672, 1.1128400677679673, 1.113450272757061, 1.114060555110572, 1.1146709146611247, 1.1152813512413426, 1.1158918646838498, 1.1165024548212708, 1.1171131214862287, 1.1177238645113481, 1.1183346837292527, 1.1189455789725666, 1.119556550073914, 1.1201675968659184, 1.120778719181204, 1.1213899168523946, 1.1220011897121147, 1.1226125375929876, 1.1232239603276377, 1.1238354577486891, 1.1244470296887652, 1.1250586759804906, 1.1256703964564889, 1.1262821909493841, 1.1268940592918002, 1.1275060013163614, 1.1281180168556915, 1.1287301057424144, 1.129342267809154, 1.1299545028885345, 1.1305668108131799, 1.1311791914157141, 1.1317916445287608, 1.1324041699849445, 1.1330167676168887, 1.1336294372572178, 1.1342421787385555, 1.1348549918935256, 1.1354678765547526, 1.1360808325548599, 1.136693859726472, 1.1373069579022124, 1.1379201269147052, 1.1385333665965749, 1.1391466767804448, 1.1397600572989393, 1.1403735079846817, 1.140987028670297, 1.1416006191884085, 1.1422142793716403, 1.1428280090526164, 1.1434418080639608, 1.1440556762382974, 1.1446696134082504, 1.1452836194064435, 1.1458976940655008, 1.1465118372180461, 1.1471260486967036, 1.1477403283340972, 1.1483546759628509, 1.1489690914155888, 1.1495835745249345, 1.1501981251235125, 1.1508127430439463, 1.1514274281188599, 1.1520421801808776, 1.1526569990626232, 1.1532718845967207, 1.1538868366157939, 1.1545018549524673, 1.1551169394393641, 1.155732089909109, 1.1563473061943255, 1.1569625881276377, 1.1575779355416698, 1.1581933482690454, 1.1588088261423886, 1.1594243689943236, 1.160039976657474, 1.1606556489644642, 1.161271385747918, 1.1618871868404592, 1.1625030520747119, 1.1631189812833, 1.1637349742988476, 1.1643510309539788, 1.1649671510813173, 1.1655833345134874, 1.1661995810831125, 1.1668158906228172, 1.1674322629652252, 1.1680486979429603, 1.1686651953886469, 1.1692817551349086, 1.1698983770143694, 1.1705150608596535, 1.1711318065033849, 1.1717486137781874, 1.1723654825166849, 1.1729824125515014, 1.1735994037152611, 1.1742164558405879, 1.1748335687601057, 1.1754507423064382, 1.1760679763122099, 1.1766852706100446, 1.1773026250325662, 1.1779200394123985, 1.1785375135821661, 1.1791550473744921, 1.179772640622001, 1.1803902931573167, 1.1810080048130633, 1.1816257754218646, 1.1822436048163447, 1.1828614928291272, 1.1834794392928365, 1.1840974440400966, 1.1847155069035313, 1.1853336277157644, 1.1859518063094203, 1.1865700425171226, 1.1871883361714954, 1.1878066871051627, 1.1884250951507485, 1.1890435601408766, 1.1896620819081714, 1.1902806602852565, 1.1908992951047559, 1.1915179861992937, 1.1921367334014938, 1.1927555365439801, 1.1933743954593767, 1.1939933099803077, 1.1946122799393968, 1.1952313051692682, 1.1958503855025457, 1.1964695207718534, 1.1970887108098152, 1.1977079554490551, 1.198327254522197, 1.1989466078618651, 1.1995660153006831, 1.2001854766712752, 1.2008049918062651, 1.201424560538277, 1.2020441826999351, 1.2026638581238627, 1.2032835866426843, 1.2039033680890239, 1.2045232022955052, 1.2051430890947523, 1.2057630283193892, 1.20638301980204, 1.2070030633753284, 1.2076231588718784, 1.2082433061243141, 1.2088635049652596, 1.2094837552273385, 1.2101040567431753, 1.2107244093453935, 1.2113448128666173, 1.2119652671394707, 1.2125857719965774, 1.2132063272705618, 1.2138269327940474, 1.2144475883996586, 1.2150682939200192, 1.2156890491877532, 1.2163098540354844, 1.2169307082958372, 1.2175516118014351, 1.2181725643849024, 1.2187935658788629, 1.2194146161159405, 1.2200357149287595, 1.2206568621499436, 1.2212780576121169, 1.2218993011479033, 1.2225205925899267, 1.2231419317708114, 1.2237633185231811, 1.2243847526796596, 1.2250062340728713, 1.2256277625354401, 1.2262493378999899, 1.2268709599991443, 1.2274926286655279, 1.2281143437317641, 1.2287361050304775, 1.2293579123942915, 1.2299797656558304, 1.230601664647718, 1.2312236092025786, 1.2318455991530355, 1.2324676343317136, 1.2330897145712361, 1.2337118397042273, 1.2343340095633111, 1.2349562239811116, 1.2355784827902525, 1.2362007858233581, 1.2368231329130523, 1.2374455238919591, 1.2380679585927021, 1.2386904368479059, 1.2393129584901939, 1.2399355233521905, 1.2405581312665193, 1.2411807820658045, 1.2418034755826701, 1.24242621164974, 1.2430489900996382, 1.2436718107649885, 1.2442946734784153, 1.2449175780725421, 1.2455405243799933, 1.2461635122333925, 1.2467865414653638, 1.2474096119085314, 1.2480327233955191, 1.2486558757589505, 1.2492790688314503, 1.2499023024456422, 1.2505255764341499, 1.2511488906295976, 1.2517722448646091, 1.2523956389718087, 1.2530190727838202, 1.2536425461332676, 1.2542660588527748, 1.2548896107749656, 1.2555132017324646, 1.2561368315578951, 1.2567605000838813, 1.2573842071430472, 1.258007952568017, 1.2586317361914146, 1.2592555578458635, 1.2598794173639882, 1.2605033145784124, 1.2611272493217602, 1.2617512214266555, 1.2623752307257226, 1.262999277051585, 1.2636233602368667, 1.2642474801141921, 1.2648716365161849, 1.265495829275469, 1.2661200582246686, 1.2667443231964073, 1.2673686240233095, 1.2679929605379989, 1.2686173325730996, 1.2692417399612359, 1.269866182535031, 1.2704906601271095, 1.2711151725700951, 1.2717397196966118, 1.2723643013392836, 1.2729889173307347, 1.2736135675035887, 1.2742382516904698, 1.2748629697240021, 1.275487721436809, 1.2761125066615153, 1.2767373252307443, 1.2773621769771204, 1.2779870617332674, 1.2786119793318091, 1.2792369296053696, 1.2798619123865733, 1.2804869275080435, 1.2811119748024045, 1.2817370541022803, 1.2823621652402948, 1.2829873080490721, 1.283612482361236, 1.2842376880094106, 1.2848629248262198, 1.2854881926442876, 1.286113491296238, 1.286738820614695, 1.2873641804322826, 1.2879895705816244, 1.2886149908953448, 1.2892404412060678, 1.2898659213464172, 1.290491431149017, 1.2911169704464913, 1.2917425390714636, 1.2923681368565587, 1.2929937636343998, 1.2936194192376111, 1.2942451034988169, 1.2948708162506408, 1.2954965573257069, 1.2961223265566393, 1.2967481237760619, 1.2973739488165985, 1.2979998015108731, 1.29862568169151, 1.2992515891911329, 1.2998775238423659, 1.3005034854778328, 1.3011294739301578, 1.3017554890319647, 1.3023815306158775, 1.3030075985145202, 1.3036336925605168, 1.3042598125864913, 1.3048859584250676, 1.3055121299088697, 1.3061383268705216, 1.3067645491426474, 1.3073907965578708, 1.308017068948816, 1.3086433661481067, 1.3092696879883672, 1.3098960343022215, 1.3105224049222932, 1.3111487996812066, 1.3117752184115854, 1.3124016609460538, 1.3130281271172359, 1.3136546167577552, 1.3142811297002364, 1.3149076657773027, 1.3155342248215784, 1.3161608066656876, 1.3167874111422542, 1.3174140380839021, 1.3180406873232553, 1.3186673586929378, 1.3192940520255736, 1.3199207671537865, 1.3205475039102008, 1.3211742621274403, 1.3218010416381287, 1.3224278422748905, 1.3230546638703493, 1.3236815062571294, 1.3243083692678543, 1.3249352527351486, 1.3255621564916356, 1.3261890803699397, 1.3268160242026847, 1.3274429878224947, 1.3280699710619936, 1.3286969737538055, 1.3293239957305543, 1.3299510368248637, 1.3305780968693581, 1.3312051756966614, 1.3318322731393972, 1.3324593890301899, 1.3330865232016633, 1.3337136754864414, 1.3343408457171484, 1.3349680337264078, 1.3355952393468438, 1.3362224624110803, 1.3368497027517416, 1.3374769602014516, 1.3381042345928338, 1.3387315257585126, 1.339358833531112, 1.3399861577432557, 1.3406134982275679, 1.3412408548166723, 1.3418682273431934, 1.3424956156397547, 1.3431230195389803, 1.3437504388734942, 1.3443778734759204, 1.3450053231788828, 1.3456327878150056, 1.3462602672169124, 1.3468877612172276, 1.3475152696485748, 1.348142792343578, 1.3487703291348616, 1.3493978798550492, 1.3500254443367647, 1.3506530224126323, 1.351280613915276, 1.3519082186773197, 1.3525358365313871, 1.3531634673101027, 1.3537911108460901, 1.3544187669719734, 1.3550464355203764, 1.3556741163239234, 1.3563018092152384, 1.3569295140269448, 1.3575572305916672, 1.3581849587420292, 1.358812698310655, 1.3594404491301684, 1.3600682110331934, 1.3606959838523542, 1.3613237674202747, 1.3619515615695785, 1.36257936613289, 1.363207180942833, 1.3638350058320317, 1.3644628406331096, 1.3650906851786913, 1.3657185393014002, 1.3663464028338606, 1.3669742756086962, 1.3676021574585313, 1.3682300482159895, 1.3688579477136953, 1.3694858557842724, 1.3701137722603447, 1.3707416969745363, 1.3713696297594709, 1.3719975704477729, 1.372625518872066, 1.3732534748649741, 1.3738814382591213, 1.3745094088871319, 1.3751373865816294, 1.375765371175238, 1.3763933625005813, 1.3770213603902839, 1.3776493646769694, 1.378277375193262, 1.3789053917717853, 1.3795334142451634, 1.3801614424460205, 1.3807894762069806, 1.3814175153606674, 1.382045559739705, 1.3826736091767171, 1.3833016635043283, 1.3839297225551621, 1.3845577861618426, 1.3851858541569937, 1.3858139263732396, 1.3864420026432038, 1.3870700827995108, 1.3876981666747845, 1.3883262541016486, 1.3889543449127273, 1.3895824389406444, 1.3902105360180239, 1.39083863597749, 1.3914667386516664, 1.3920948438731773, 1.3927229514746466, 1.3933510612886981, 1.3939791731479561, 1.3946072868850443, 1.3952354023325868, 1.3958635193232074, 1.3964916376895304, 1.3971197572641794, 1.3977478778797789, 1.3983759993689522, 1.3990041215643239, 1.3996322442985174, 1.4002603674041572, 1.400888490713867, 1.4015166140602711, 1.4021447372759928, 1.4027728601936564, 1.4034009826458862, 1.4040291044653059, 1.4046572254845395, 1.4052853455362109, 1.4059134644529441, 1.4065415820673635, 1.4071696982120923, 1.407797812719755, 1.4084259254229754, 1.4090540361543775, 1.4096821447465855, 1.410310251032223, 1.4109383548439141, 1.4115664560142831, 1.4121945543759535, 1.4128226497615495, 1.413450742003695, 1.4140788309350141, 1.414706916388131, 1.4153349981956689, 1.4159630761902524, 1.4165911502045054, 1.4172192200710518, 1.4178472856225155, 1.4184753466915208, 1.4191034031106911, 1.4197314547126507, 1.4203595013300239, 1.4209875427954342, 1.4216155789415057, 1.4222436096008624, 1.4228716346061283, 1.4234996537899272, 1.4241276669848835, 1.4247556740236207, 1.4253836747387632, 1.4260116689629347, 1.4266396565287591, 1.4272676372688606, 1.4278956110158632, 1.4285235776023906, 1.4291515368610668, 1.4297794886245161, 1.4304074327253624, 1.4310353689962292, 1.4316632972697412, 1.4322912173785218, 1.432919129155195, 1.4335470324323853, 1.4341749270427164, 1.4348028128188119, 1.4354306895932964, 1.4360585571987934, 1.4366864154679271, 1.4373142642333212, 1.4379421033276001, 1.4385699325833874, 1.4391977518333074, 1.4398255609099837, 1.4404533596460407, 1.4410811478741024, 1.4417089254267919, 1.4423366921367342, 1.4429644478365529, 1.4435921923588719, 1.4442199255363153, 1.4448476472015068, 1.4454753571870709, 1.4461030553256311, 1.4467307414498114, 1.4473584153922361, 1.4479860769855293, 1.4486137260623142, 1.4492413624552154, 1.4498689859968565, 1.450496596519862, 1.4511241938568555, 1.4517517778404609, 1.4523793483033023, 1.4530069050780039, 1.4536344479971892, 1.4542619768934828, 1.4548894915995081, 1.4555169919478894, 1.4561444777712504, 1.4567719489022153, 1.4573994051734083, 1.4580268464174528, 1.4586542724669731, 1.4592816831545932, 1.4599090783129369, 1.4605364577746283, 1.4611638213722915, 1.4617911689385503, 1.4624185003060288, 1.4630458153073507, 1.4636731137751404, 1.4643003955420215, 1.4649276604406181, 1.4655549083035544, 1.466182138963454, 1.4668093522529411, 1.4674365480046396, 1.4680637260511735, 1.4686908862251666, 1.4693180283592433, 1.4699451522860274, 1.4705722578381426, 1.471199344848213, 1.4718264131488628, 1.4724534625727159, 1.4730804929523962, 1.4737075041205276, 1.474334495909734, 1.4749614681526397, 1.4755884206818686, 1.4762153533300442, 1.4768422659297911, 1.4774691583137329, 1.4780960303144939, 1.478722881764698, 1.4793497124969686, 1.4799765223439305, 1.4806033111382071, 1.4812300787124228, 1.481856824899201, 1.4824835495311663, 1.4831102524409423, 1.4837369334611532, 1.484363592424423, 1.4849902291633752, 1.4856168435106341, 1.4862434352988236, 1.486870004360568, 1.4874965505284909, 1.4881230736352165, 1.4887495735133687, 1.4893760499955715, 1.4900025029144488, 1.4906289321026245, 1.4912553373927229, 1.4918817186173674, 1.4925080756091826, 1.4931344082007922, 1.4937607162248201, 1.4943869995138903, 1.495013257900627, 1.4956394912176538, 1.4962656992975951, 1.4968918819730745, 1.4975180390767162, 1.4981441704411442, 1.4987702758989823, 1.4993963552828544, 1.5000224084253848, 1.5006484351591973, 1.5012744353169158, 1.5019004087311645, 1.5025263552345671, 1.5031522746597479, 1.5037781668393306, 1.5044040316059393, 1.5050298687921979, 1.5056556782307302, 1.5062814597541607, 1.5069072131951131, 1.5075329383862111, 1.5081586351600791, 1.5087843033493407, 1.5094099427866201, 1.5100355533045413, 1.5106611347357282, 1.5112866869128045, 1.5119122096683948, 1.512537702835123, 1.5131631662456124, 1.5137885997324876, 1.5144140031283722, 1.5150393762658902]), normalise([0.95067386296337808, 0.95097555091199182, 0.96280225180417267, 0.96757147991833614, 0.97827703737750338, 0.99172548287252427, 0.98336876823080388, 0.9995729992544311, 1.0, 0.99334862308810012, 0.97783025344824892, 0.96036222456272446, 0.94086677832625343, 0.91148127470843909, 0.88433998874111464, 0.86736370152252595, 0.84152162416848042, 0.80496330428419827, 0.7914065018025731, 0.78105142470219546, 0.75990049947642435, 0.74515775277583329, 0.74876148329820102, 0.78478049671438743, 0.83078844598468438, 0.87290600107015437, 0.90377910989863142, 0.9240058487465157, 0.93718882171219908, 0.93649419526325317, 0.92866787644958082, 0.92044789428876339, 0.91289809639030861, 0.90768129154216171, 0.90269579678478518, 0.9004788474318961, 0.91165123473151788, 0.91750082412549061, 0.92937959950752924, 0.93847508954584546, 0.9338528393108122, 0.93032977047260212, 0.90867969640237189, 0.89861041995170809, 0.9009141551009715, 0.89088117093087138, 0.86990224332059907, 0.88136121880277696, 0.89737598503733851, 0.89049374786316393, 0.8845113998760622, 0.88108127721243346, 0.85802930241290887, 0.83343674626245789, 0.81158606951263734, 0.78406575674023316, 0.74940569541115354, 0.7225645717989434, 0.69610401677489564, 0.68592599868942217, 0.70747060871105982, 0.73785835337693928, 0.7683844183382581, 0.79729553888982596, 0.81431139804578401, 0.83902883272350137, 0.87480145715085245, 0.88913815307174193, 0.88844895447852823, 0.88770801594967064, 0.87103711697135733, 0.84643550150378266, 0.82612744808096805, 0.80419549706255999, 0.78167906594031633, 0.77312814924807094, 0.78423337226118583, 0.78715600605862135, 0.7788733671669158, 0.77590699700014731, 0.80364794237865855, 0.79846999477097025, 0.82377910975851953, 0.81953168847632074, 0.83433297824812258, 0.83990719253847368, 0.86494299504011685, 0.89759952259848064, 0.89358040202255518, 0.87003993761577814, 0.85071726218577448, 0.8280474069533702, 0.80796133346635146, 0.77424884392294779, 0.76818309443753896, 0.75212576559401301, 0.75896741461832939, 0.79591449137935055, 0.78576020756671017, 0.7823807328452157, 0.7950027187214842, 0.79082851509538432, 0.76852523317950583, 0.76419759793386222, 0.77405889053018917, 0.74894143783313427, 0.73313421535526679, 0.74386194172123354, 0.72754875938198116, 0.67078866192863174, 0.64040869093430697, 0.63586686730632846, 0.624948575522559, 0.5965597907399619, 0.60132246038764936, 0.62665914697277769, 0.6507369064748082, 0.6880263597289703, 0.71908935108801286, 0.74247203323212707, 0.74712077642175456, 0.74411016462278257, 0.73781383656204058, 0.75705149919615555, 0.75460086758270184, 0.75506686802099343, 0.74429183460309634, 0.74727308356014788, 0.75870308524847652, 0.76224105822457822, 0.75870364965202586, 0.76260643700024411, 0.76402784630494558, 0.75226035377343503, 0.7509223029093699, 0.76242720076888837, 0.76324984715325062, 0.7664463383908191, 0.76268632587337482, 0.75499405042322276, 0.75016402333378185, 0.75166234176095492, 0.75803473937780663, 0.75105937422206581, 0.73717232479810468, 0.72600463713961927, 0.72722247175722965, 0.72901346198278982, 0.73273964479152764, 0.74581354391113908, 0.74975475887382148, 0.7460269456797346, 0.75366512291240562, 0.74542043075435838, 0.74088990307836045, 0.73576442911699269, 0.73854373743647739, 0.74799586197939716, 0.75643553274352437, 0.74228765345501901, 0.7350025695297322, 0.73487301722970744, 0.73548444052076589, 0.73254799378602231, 0.74785646614806911, 0.73956723000316371, 0.72983168299265277, 0.72717901614328917, 0.73080883936974528, 0.73963132633382656, 0.7298577440133398, 0.71584973940833796, 0.71039545405988602, 0.70957322979817494, 0.70331224935994074, 0.69297437553804364, 0.70482704536649632, 0.70458552751512638, 0.71171923949021021, 0.71414593008883565, 0.71226572575974478, 0.69481810709843383, 0.67403396507947821, 0.66351456492393224, 0.66692042245792926, 0.66397311067647324, 0.67266854416424848, 0.66885879781121105, 0.67469177763810428, 0.65315174008972599, 0.64486560350640754, 0.63419442620472311, 0.61363464021932379, 0.592226640178384, 0.57147784045889904, 0.5665598959214253, 0.55328103432689146, 0.53872447773593279, 0.51930907812655525, 0.52397601022964135, 0.5433417786363145, 0.54687853672846876, 0.57622161095578506, 0.59758781575432518, 0.60518924314766787, 0.61302347184642136, 0.62265370494499006, 0.62590246487277779, 0.64134780050690676, 0.64393518829388674, 0.64048305288617102, 0.63687848130620361, 0.63576606786978651, 0.62975753640229482, 0.63224282443477986, 0.63775488282720483, 0.64228866129962658, 0.64454406157851607, 0.64093376133346658, 0.63219878123736528, 0.62288453537858335, 0.61777152481009989, 0.61916552459960394, 0.62475019632468365, 0.6307387034361579, 0.63406958528011637, 0.63409325444670284, 0.63141443862803748, 0.62685097092233943, 0.62194584657136087, 0.61895529341107436, 0.62005940953187499, 0.62569037876923861, 0.63266538839755593, 0.63652429870139982, 0.63402754753280766, 0.62633589226574982, 0.61713113419973387, 0.6100676617671722, 0.60643997758513246, 0.60454928907025318, 0.60234507749234201, 0.59876212261995387, 0.59488767451744395, 0.59261862078402316, 0.59317877928550877, 0.59509260697540345, 0.59523515526606763, 0.59070677862253207, 0.58188616097913637, 0.57352796851273158, 0.57080462987451508, 0.5759598275741894, 0.58455557522154666, 0.58953331920956353, 0.58540468585133665, 0.57384242703784172, 0.5609726690375928, 0.55296886547383817, 0.55234056127052245, 0.55640738218573671, 0.56167271170383193, 0.56547872181926928, 0.56701120755622481, 0.566103102055869, 0.56293021320433356, 0.55865946276872791, 0.55507334967082012, 0.5538546861527841, 0.555215558346703, 0.5573202850732929, 0.5580855190012961, 0.55635183675941779, 0.55303720411393331, 0.55017193160777167, 0.55632505949550048, 0.55842617422142993, 0.55930398380703561, 0.56111792194745902, 0.56311362917288199, 0.56377839855324108, 0.56159729129614877, 0.5562118661134865, 0.54984521462442182, 0.54571557809013993, 0.54615022745155195, 0.55003763261149563, 0.55423861460588042, 0.55564334910520108, 0.55321557297709312, 0.54853670250814357, 0.54348669700697827, 0.53933052607900034, 0.53609075651280125, 0.53417511289912745, 0.54363952542320537, 0.54489689439772104, 0.54371627897472052, 0.54254577771331125, 0.54139391969249884, 0.54026102780534135, 0.5391495659917539, 0.5380678682672404, 0.53701966314005223, 0.53600561600826757, 0.53503308134642835, 0.53410377702089928, 0.53322050899932705, 0.5323889944583029, 0.53160153675102861, 0.53084435997937129, 0.53010040628256294, 0.52935099099189609, 0.52858527165141311, 0.52778440009470673, 0.52693157459486761, 0.52601066592051859, 0.52500786972009084, 0.52390384435669779, 0.52268499027723325, 0.52133420049297019, 0.51983766608082482, 0.51817447467771383, 0.51633010764315568, 0.51428908928419848, 0.51205004173409818, 0.50962868863656563, 0.50705445099604163, 0.50435853336671255, 0.50157171258705469, 0.49872378913383958, 0.49584595944071141, 0.49296791807543189, 0.49012166726388473, 0.48733383206987241, 0.48463705253549344, 0.48206268658987744, 0.47964380208939272, 0.47740591236968971, 0.47538082710780466, 0.47359943963300771, 0.47208944167697486, 0.47085482184063782, 0.46987718011231366, 0.46913151363975825, 0.46859981385250715, 0.46826056877927347, 0.46809150706272662, 0.46807051423786339, 0.46817389277577715, 0.4683828811748339, 0.46867809945712413, 0.46903257913684149, 0.46942700028689333, 0.46983817386822357, 0.47024502140739316, 0.47062745388381477, 0.47096384565751759, 0.47123519968296002, 0.47144172289278274, 0.47158355145177938, 0.47165836227662283, 0.47167459650427029, 0.47162727968604001, 0.47152077228781786, 0.4713556274379429, 0.47113320698791356, 0.47085723155185816, 0.47052669191136981, 0.47013997533060337, 0.46970328545047041, 0.46921587685456462, 0.46868028428702324, 0.46809675759294783, 0.46746766746024915, 0.46679372731551366, 0.4660808566342331, 0.4653309233485094, 0.4645481835128249, 0.46373613352170273, 0.46289954968563635, 0.46203851150604325, 0.46115977169533545, 0.46026724403582148, 0.45936242317851389, 0.45844793778008275, 0.45752915474685829, 0.45661241825314841, 0.45569790990513565, 0.45478805140170597, 0.45388921686076783, 0.45300193302022751, 0.4521257747508054, 0.45126117720564929, 0.45040380008805775, 0.44955638117628938, 0.44871716318144994, 0.447883595599044, 0.44705769396207307, 0.44623451532227415, 0.44541618649898512, 0.44460021001039512, 0.44378470574702, 0.44297237649639609, 0.4421588071600342, 0.44134478095041002, 0.44052917595732433, 0.43970896990563224, 0.43888794640793205, 0.43806474448674715, 0.43723890198829229, 0.43641039886988359, 0.43558302543121585, 0.43475297450838885, 0.43392499698246184, 0.43309617971572684, 0.4322684739269389, 0.43143959266303389, 0.43061316103250397, 0.4297870067055593, 0.42896463519244954, 0.4281451101147668, 0.42732662949866002, 0.42651289560506961, 0.425700097594579, 0.42489280214541714, 0.42408659495027839, 0.4232855470312048, 0.42248486758278009, 0.42168734193263269, 0.42089182029062655, 0.42010113738931282, 0.41931057168347752, 0.41852406614689225, 0.41773722497172638, 0.41695463124051313, 0.41617475636527157, 0.41539482130415506, 0.41461568676555749, 0.41383967249300885, 0.41306394717859396, 0.41229101807598417, 0.41151942384146456, 0.41074774573944245, 0.4099779303549847, 0.40921046330057659, 0.40844347578124329, 0.40767969554636097, 0.40691637505673767, 0.40615311204219701, 0.40539278830969222, 0.40463361174206175, 0.40387580932871797, 0.40312078527402989, 0.4023657572851177, 0.40161135046383978, 0.40086013884589189, 0.40010930046215876, 0.39936085172269659, 0.39861196518082675, 0.39786750899066448, 0.39712261736770066, 0.39637790857658983, 0.3956368918681587, 0.3948958202196024, 0.39415721560300526, 0.39341891691653619, 0.39268344762043605, 0.39194930564410385, 0.39121508927168769, 0.39048281219767567, 0.38975284095505347, 0.3890234052537771, 0.38829719252756556, 0.38757140459426659, 0.38684571922899125, 0.38612305235245087, 0.38540240971527745, 0.38468329644163224, 0.38396623260695273, 0.38325050849985265, 0.38253668392591689, 0.38182493875640888, 0.38111602954714741, 0.3804074416729758, 0.37970270639648052, 0.37900081992983875, 0.37829825547297147, 0.37760063897842105, 0.37690462302337058, 0.37620963122319551, 0.37551944112365904, 0.37482962024057892, 0.37414147811122789, 0.37345677385471304, 0.37277260711843391, 0.37209140317787059, 0.37141119121085436, 0.37073221982921312, 0.37005461273185469, 0.36937984090984649, 0.36870509093957737, 0.36803099649907689, 0.36736012647235472, 0.36668960199724104, 0.36601926723376388, 0.36535144616702953, 0.36468237487076338, 0.364017662791425, 0.36335299843955954, 0.36268705152883296, 0.36202435505891506, 0.36136217526966941, 0.36070008389079256, 0.36004053883276071, 0.35937976234570296, 0.35872148299556633, 0.35806353470945662, 0.35740811969995884, 0.35675151990078946, 0.35609595533731148, 0.35543993363371817, 0.35478632525498743, 0.35413518807071281, 0.35348284837230892, 0.3528330522911402, 0.35218358504539177, 0.35153663191260032, 0.35088844606016989, 0.3502446704851076, 0.34960106493272097, 0.3489594200960579, 0.34832100735777222, 0.34768171806783421, 0.3470473706300477, 0.34641484328401634, 0.34578324998350007, 0.34515479811980809, 0.34452667698928657, 0.34390154627830405, 0.34327851079315774, 0.34265865916655369, 0.34203915517643441, 0.34142257672804338, 0.3408072911716048, 0.34019338913468955, 0.33958234778739238, 0.33897167636812076, 0.33836346945145496, 0.33775409193647971, 0.33714732323926377, 0.33654315752644276, 0.33593792905548531, 0.33533506102751043, 0.33473262497214507, 0.33413264670610593, 0.33353167320933236, 0.33293300083558491, 0.33233476548736829, 0.33173648450120485, 0.33113815803823465, 0.33054088959511752, 0.32994493413174147, 0.32934892910144836, 0.32875206032806253, 0.32815736186544442, 0.32756319996957184, 0.32696899405338242, 0.32637474427682228, 0.32578045079983781, 0.32518774540642365, 0.32459543970944543, 0.32400421045815797, 0.32341418009744183, 0.3228256773926273, 0.32224220090721684, 0.32165860411571318, 0.3210802737445248, 0.32050484754896902, 0.31993235056117475, 0.31936323115104226, 0.31879955533056126, 0.3182413243489523, 0.31768862777844792, 0.31714146222808809, 0.31659974532300661, 0.31606347827349307, 0.31553049764033275, 0.31499757818191265, 0.3144646224534956, 0.31393163059831675, 0.31339439651124362, 0.31285215133883959, 0.31230519396328749, 0.31174810251140073, 0.31118192404423661, 0.31060660291661629, 0.31001701112039776, 0.30941421451745132, 0.30879519351160967, 0.30815927941400578, 0.30750681288196791, 0.30683902267907176, 0.30616358408031386, 0.30548530415516578, 0.30480977818506516, 0.30414422124235718, 0.30349358291875095, 0.30286509742459317, 0.30226044449668887, 0.30169007093549105, 0.3011571825869756, 0.30066942987551459, 0.30023276329073278, 0.2998500468365134, 0.29953030278449932, 0.29927826695829268, 0.29909371497760395, 0.29896530510542346, 0.29886901761632906, 0.2987811275623653, 0.29867871737346885, 0.29854108785825967, 0.29833992355923777, 0.29805376504467546, 0.29765950219387327, 0.29713634927105637, 0.29645888939736909, 0.2956006872566001, 0.29454017424738405, 0.29325698760714597, 0.29172547989121683, 0.28991770960325003, 0.28782735575272611, 0.28547684082884284, 0.2829139340288106, 0.28018083898100421, 0.27733221093285226, 0.2744153692253552, 0.27147563353409848, 0.26856543803167604, 0.26573081144385019, 0.26302105550504673, 0.26048362282928861, 0.25817116899825532, 0.25612825079580487, 0.25440029506252104, 0.25303973961428039, 0.25210194857245044, 0.251611085080435, 0.25155546260873585, 0.25188996947472359, 0.25257653750136649, 0.25356367366235083, 0.25480173729645123, 0.2562459443181298, 0.25785292942326715, 0.25957457132404133, 0.26136234935081204, 0.26317414908699138, 0.26496148892675014, 0.26667538189649798, 0.26827042757173597, 0.26970448317288537, 0.27092453056486471, 0.27189830328354619, 0.27262168534553388, 0.27311486811224794, 0.27339476682505987, 0.27348113631159299, 0.27340015228817738, 0.27316523110249158, 0.27279658042125521, 0.27231813973873775, 0.27174986047812605, 0.27110985743226956, 0.27041848312127686, 0.26969535278110823, 0.26896000609079662, 0.26823273095467776, 0.26753628691985304, 0.2668844208836379, 0.26628130220058188, 0.26572324115342483, 0.26520771061639381, 0.2647299513199397, 0.2642820188372047, 0.26386194695776816, 0.26346958518883068, 0.26309421286205265, 0.26273442709141609, 0.26238750289176971, 0.26204635729030062, 0.26170692498703746, 0.26136629267604333, 0.26102051887932975, 0.26066580573163028, 0.26029674120017809, 0.25991435631423954, 0.25951766817452249, 0.25910971337568534, 0.25869320117339245, 0.25826568379911913, 0.25783269026380046, 0.2573916943210045, 0.25694511743038978, 0.25649425002761583, 0.25604086966307515, 0.25558506169486156, 0.25512994683644569, 0.25467683405923014, 0.25422326003663531, 0.25377509831944928, 0.25333247928534397, 0.25289404859130099, 0.25245822742064361, 0.2520276149880058, 0.25159925311079834, 0.25117558914515242, 0.25075527007140364, 0.2503353880891005, 0.24991879787934551, 0.24950444424994161, 0.24909086568281136, 0.24867848718902189, 0.24826998341983361, 0.24785894979269385, 0.24744767379936328, 0.24703898872589858, 0.24662832739021823, 0.24621690655894463, 0.24580547351425067, 0.2453940283659504, 0.2449825712238575, 0.24457267855734605, 0.24416318711023291, 0.24375169464564583, 0.24334114092544412, 0.24293220104616906, 0.24252305633485235, 0.2421132218061712, 0.24170550171011204, 0.24130009291350116, 0.24089505244431383, 0.24049124223212234, 0.24008859739309468, 0.23968595912483906, 0.23928571472919771, 0.2388847107674841, 0.23848700847617998, 0.23809126980408132, 0.23769552460230151, 0.23729894229191134, 0.23690462670257478, 0.23651268356761607, 0.23612061454216449, 0.23572742713892711, 0.23533758449415942, 0.23494823935424514, 0.23455914012578502, 0.23417261659020513, 0.23378371821077423, 0.23339634658245517, 0.23301142873184119, 0.23262650910740548, 0.23224158155624977, 0.23185664618102639, 0.23147201044226626, 0.2310899118904807, 0.23070671247291924, 0.23032403508776766, 0.22994205003026474, 0.22956154637986378, 0.22918103570658224, 0.22880051811187851, 0.22841999369721128, 0.22803946256403904, 0.22766040087951012, 0.22728205390142586, 0.22690421562242341, 0.22652528273209849, 0.2261488843230996, 0.22577279229556774, 0.22539669433042631, 0.22502059052793993, 0.22464609117557732, 0.2242744018871102, 0.22390275880171395, 0.22353111034475032, 0.22316113717436301, 0.22279391611777824, 0.22242673001978333, 0.22206057794633183, 0.22169589682784677, 0.22133153575423598, 0.22097161453442121, 0.22061054827245391, 0.22025208892474313, 0.21989382702236304, 0.2195361548453644, 0.21918067955543039, 0.21882445279110302, 0.21846811466713284, 0.21811330699444767, 0.21775950032770264, 0.21740573756597653, 0.21705449427944387, 0.21670162535942472, 0.21634780521985147, 0.21599598894021785, 0.21564372135028909, 0.21529062619767264, 0.21493951376630141, 0.21458794301333944, 0.21423824625610866, 0.2138860295520347, 0.21353629323143478, 0.21318692734419356, 0.21283755920206673, 0.21248902657120039, 0.21214230685765287, 0.2117953345528957, 0.21145002856272616, 0.21110316994708347, 0.21076057737281911, 0.21041586847875379, 0.2100741610831228, 0.20973373540708601, 0.20939330844113963, 0.20905288027599947, 0.20871293912120154, 0.20837635153514719, 0.20804039967485793, 0.20770444698924262, 0.20736849356782347, 0.20703360786090313, 0.20670189653409582, 0.20636796627172985, 0.20603597556548514, 0.20570730255167533, 0.2053760451153776, 0.20504707460565011, 0.204717246793301, 0.20438974735234336, 0.20406275031942786, 0.20373601604441136, 0.20341186182407023, 0.20308658760925097, 0.20276194832137925, 0.20243785054601415, 0.20211461764259714, 0.201789374095104, 0.20146674248770155, 0.20114423011093069, 0.20082248791242768, 0.20050267541431957, 0.20018026511179643, 0.19985811897109973, 0.1995384685415024, 0.19921777963188866, 0.19889747985283918, 0.19857808632932047, 0.19826006532757118, 0.19794204608694435, 0.19762402869218754, 0.19730601322804872, 0.19698893183948604, 0.19667350938306433, 0.19635802161300567, 0.19604449368915597, 0.19573195735551097, 0.19542098363494056, 0.19511005363940398, 0.19480101433220348, 0.19449197764248127, 0.19418296250408737, 0.19387656271899192, 0.19357145445532156, 0.19326409771450828, 0.19295938434889262, 0.19265733034695262, 0.19235384421534418, 0.19205047838959724, 0.1917483701908359, 0.19144663818071156, 0.19114659231213144, 0.19084531944767522, 0.19054037300656265, 0.19013218141889002, 0.19011945902664409, 0.19026203258239185, 0.19030955339848615, 0.19011057941879253, 0.18962834680275281, 0.18890730886540424, 0.18791484703041078, 0.18657211041755081, 0.18502865681511882, 0.18395062363702219, 0.18403196326157065, 0.18489119900769874, 0.18569681959590659, 0.18610345288225782, 0.18600575670761282, 0.18529458899145493, 0.18407169459174036, 0.18285974027704163, 0.18223220700919765, 0.18227897525057749, 0.18287292060899257, 0.18371484543620586, 0.18435052469501034, 0.18414127327747412, 0.18309984290084236, 0.18198319893632528, 0.18121889110060271, 0.18065233372951559, 0.18004600823899039, 0.17928617298167049, 0.17684138546251005, 0.17628111953622413, 0.17785924478004031, 0.17948090210622608, 0.18016242250566339, 0.1799031266807242, 0.17890641207865954, 0.17753450922312206, 0.17661286540832177, 0.17697488585429297, 0.1785625225486383, 0.18025687113766678, 0.18086536137244297, 0.17993986777897675, 0.17759769622683358, 0.17446339758582308, 0.17167872858199046, 0.17033348998230252, 0.17064186953257598, 0.17186146514242751, 0.17303603414531255, 0.17377950614087834, 0.17427307424580613, 0.17490640855537357, 0.17587432085113597, 0.17686280708401098, 0.17718138236909151, 0.1763854033736458, 0.17483956295444256, 0.17320755327930923, 0.17201695055044691, 0.17153873113374993, 0.17180536532107213, 0.17255663772097324, 0.17316898816882101, 0.17297686215610902, 0.17194508712168549, 0.17042733502076748, 0.1686423675644394, 0.16671837916915197, 0.16501957971216336, 0.16431408134243464, 0.16543059906579172, 0.16823973394902964, 0.17121684990477445, 0.17246754013742382, 0.17148875233295824, 0.16855609984246317, 0.16503015883835453, 0.16283340507498029, 0.16329215429738966, 0.16587832161352459, 0.16854079239444347, 0.16957793779630248, 0.16838275618449727, 0.16548373110544409, 0.16250984153910344, 0.16142802838539139, 0.16325455755501075, 0.16692283392901228, 0.16994678859310205, 0.17048029717322183, 0.16816850819500032, 0.16411013030638466, 0.16029939517352718, 0.15825123532380289, 0.15842082340311053, 0.160024599008289, 0.1616503111535679, 0.16230658189473809, 0.16188918810866365, 0.16116769666742919, 0.16115443018277312, 0.16226280121471365, 0.16402764187515514, 0.16534272696807659, 0.16520351356129903, 0.16338508216253955, 0.16066371434623247, 0.15845942088887327, 0.15796131441911296, 0.15943760271191068, 0.15844455182486797, 0.15455720485059152, 0.14992723747471548, 0.14660900147839576, 0.14606008931913605, 0.14835726995150908, 0.15206426509231963, 0.15526209088169932, 0.1552822213491985, 0.15376687792511146, 0.15325391941146979, 0.15370672022652504, 0.15473663319527947, 0.15572208734461965, 0.15611835851884723, 0.15574511832445817, 0.15482893539736317, 0.15389878431604836, 0.15343468666727614, 0.15360140639857167, 0.15416340194696695, 0.15457976082160571, 0.15441948622844215])
# plt.plot(esa_a0_w, esa_a0_s, label='ESA A0V', alpha=0.7)
# esa_trend = normalise(specs[0]) - rebin_spec(esa_a0_w, esa_a0_s, wav) # correct for Earth's atmosphere
# # plt.plot(wav, esa_trend, label='Trend-ESA')
# esa_lp_spec = normalise(rebin_spec(wavs[1], specs[1], wav)) - esa_trend
# plt.plot(wav, esa_lp_spec, label='LP - ESA')

# for teff in range(3600, 3701, 200):
#     m1v_w, m1v_s = load_model_spec('data/'+str(teff)+'_5.0.fits', wav_ranges[gr])  # M1V model
#     m1v_s_bin = rebin_spec(m1v_w, m1v_s, wav)
#     plt.plot(wav, m1v_s_bin, label=teff, alpha=0.8)

# for teff in range(9000, 10201, 400):
#     m1v_w, m1v_s = load_model_spec('data/'+str(teff)+'_5.0.fits', wav_ranges[gr])  # M1V model
#     m1v_s_bin = rebin_spec(m1v_w, m1v_s, wav)
#     plt.plot(wav, m1v_s_bin, label=teff, alpha=0.7)

# Earth's transmission spectrum
# earth_file = np.genfromtxt('data/mktrans_zm_10_10.dat')
# earth_wav_full, earth_spec_full = earth_file[:,0], earth_file[:,1]
# earth_s_bin = rebin_spec(earth_wav_full, earth_spec_full, wav)
# plt.plot(earth_wav_full, earth_spec_full, label='Earth', alpha=0.4)
# plt.plot(wav, earth_s_bin, label='Earth', alpha=0.4, color='g')

# plt.xlim(min(wav_hk), max(wav_hk))
# plt.ylim(0, 3)
# plt.minorticks_on()
# plt.legend()
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('Normalised flux')
# plt.show()

# with open('Ar-HD-A-lines.dat', 'w') as dat:
#     dat.write('#x_pix\twavelength\n')
#     for i in range(len(xpos)):
#         dat.write(str(xpos[i]) + '\t' + str(ang[i]) + '\n')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# hd_a_hk_spec = [0.0] * n_pix
# for j in range(s_row-ap, s_row+ap+1):   # rows to sum
#     for k in range(n_pix):          # for each pixel in row, add to pixel counts
#         hd_a_hk_spec[k] += all_data[comp]['raw_imgs'][0][j][k] / (2.0 * ap + 1.0)
#
# a0v_file = np.genfromtxt('uka0v.dat', delimiter='  ', dtype=float)
# a0v_wav_full, a0v_spec_full = a0v_file[:, 0] / 1e4, a0v_file[:, 1]
# a0v_wav, a0v_spec, a0v_wav_c, a0v_spec_c = [], [], [], []
# for i in range(len(a0v_spec_full)):
#     if wav_ranges[arc][0] < a0v_wav_full[i] < wav_ranges[arc][1]:
#         a0v_wav.append(a0v_wav_full[i])
#         a0v_spec.append(a0v_spec_full[i])
#         if i%2==0:
#             a0v_wav_c.append(a0v_wav_full[i])
#             a0v_spec_c.append(a0v_spec_full[i])
#
# plt.plot(a0v_wav_full, a0v_spec_full, label='A0V model')
# for mv in range(0,3):
#     m_file = np.genfromtxt('ukm'+str(mv)+'v.dat', delimiter='  ', dtype=float)
#     m_wav_full, m_spec_full = m_file[:, 0] / 1e4, m_file[:, 1]
#     plt.plot(m_wav_full, m_spec_full, label='M'+str(mv)+'V model')
# for line in wav_ranges[arc]+zj_range:
#     plt.axvline(line, color='k', ls='--')
# plt.legend()
# plt.show()
#
# a0v_spec /= np.nanmedian(a0v_spec)
# hd_a_hk_spec /= np.nanmedian(hd_a_hk_spec)
# hd_a_hk_spec = hd_a_hk_spec[7:] - 0.4
# y_sol = y_sol[7:]
#
# earth_file = np.genfromtxt('mktrans_zm_10_10.dat')
# earth_wav_full, earth_spec_full = earth_file[:,0], earth_file[:,1]
#
# trend = hd_a_hk_spec / a0v_spec_c[:len(hd_a_hk_spec)]
# trend /= np.nanmedian(trend)
#
# plt.plot(y_sol, hd_a_hk_spec, label='HD27267-A-HK', zorder=1)
# plt.plot(a0v_wav, a0v_spec, label='A0V model', zorder=2)
# plt.plot(earth_wav_full, earth_spec_full, label='Earth transmission', alpha=0.4, zorder=0)
# plt.plot(y_sol, trend, label='Trend', alpha=0.6, zorder=1)
# plt.xlim(min(y_sol)-0.1, max(y_sol)+0.1)
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('Normalised flux')
# plt.ylim(0, 3)
# plt.legend()
# plt.minorticks_on()
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Objects:

A:HD27267_hk_3.0 - 4                HD HK spectra - A - 3s
A:HD27267_hk_6.0 - 4                HD HK spectra - A - 6s
A:HD27267_zj_1.5 - 2                HD zJ spectra - A - 1.5s
A:HD27267_zj_4.0 - 2                HD zJ spectra - A - 4s
A:LP358_hk_12.0 - 4                 LP HK spectra - A
A:LP358_zj_25.0 - 4                 LP zJ spectra - A
Arc-Ar-Xe_hk_50.0 - 1               HK Argon + Xenon arc
Arc-Ar-Xe_zj_35.0 - 1               zJ Argon + Xenon arc
Arc-Ar-short_hk_10.0 - 2            HK Argon arc - 10s
Arc-Ar-long_hk_80.0 - 1             HK Argon arc - 80s
Arc-Ar-short_zj_3.5 - 1             zJ Argon arc - 3.5s
Arc-Ar-long_zj_15.0 - 1             zJ Argon arc - 15s
Arc-Xe-short_hk_8.0 - 1             HK Xenon arc - 8s
Arc-Xe-long_hk_80.0 - 1             HK Xenon arc - 80s
Arc-Xe-short_zj_6.0 - 1             zJ Xenon arc - 6s
Arc-Xe-long_zj_60.0 - 1             zJ Xenon arc - 60s
B:HD27267_hk_3.0 - 4                HD HK spectra - B - 3s
B:HD27267_hk_6.0 - 4                HD HK spectra - B - 6s
B:HD27267_zj_1.5 - 2                HD zJ spectra - B - 1.5s
B:HD27267_zj_4.0 - 2                HD zJ spectra - B - 4s
B:LP358_hk_12.0 - 4                 LP HK spectra - B
B:LP358_zj_25.0 - 4                 LP zJ spectra - B
DomeFlat-HK-Bright_hk_2.5 - 20      HK dome flat - bright - 2.5s
DomeFlat-HK-Bright_hk_2.7 - 1       HK dome flat - bright - 2.7s
DomeFlat-HK-Bright_hk_3.0 - 21      HK dome flat - bright - 3s
DomeFlat-HK-Dim_hk_2.5 - 20         HK dome flat - dim - 2.5s
DomeFlat-HK-Dim_hk_3.0 - 21         HK dome flat - dim - 3s
DomeFlat-zJ-Bright_zj_4.5 - 21      zJ dome flat
W-Flat-zJ-Bright_zj_1.5 - 21        zJ tungsten lamp flat

Test / acquire slit ..?
acq-slit_ar_1.0 - 2
acq-slit_ar_3.0 - 3
acq-test_ar_1.0 - 5
acq-test_ar_3.0 - 5
"""

"""
spec flat_zj_1.0
bright spec flat_hk_1.5
dim spec flat_hk_1.5
Ar short_zj_3.4
Ar off_zj_3.4
Ar long_zj_13.4
Ar off_zj_13.4
Ar short_zj_5.4
Xe off_zj_5.4
Xe long_zj_54.0
Xe off_zj_54.0
Ar short_hk_9.4
Ar off_hk_9.4
Ar long_hk_80.0
Ar off_hk_80.0
Xe short_hk_6.7
Xe off_hk_6.7
Xe long_hk_80.0
Xe off_hk_80.0
LP358_ar_5.0
through slit_ar_5.0
A:LP358 zJ_zj_25.0
B:LP358 zJ_zj_25.0
A:LP358 HK_hk_25.0
B:LP358 HK_hk_25.0
HD284647_ar_2.0
A:HD 284647 HK_hk_8.0
B:HD 284647 HK_hk_8.0
HD284647_ar_5.0
A:HD 284647 zJ_zj_10.0
B:HD 284647 zJ_zj_10.0
"""
