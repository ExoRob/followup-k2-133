import numpy as np
import requests
import pandas


def epic_lookup(k2id):
    columns = "k2_kepmag,k2_kepmagerr,k2_bjmag,k2_bjmagerr,k2_vjmag,k2_vjmagerr,k2_umag,k2_umagerr,k2_gmag," \
        "k2_gmagerr,k2_rmag,k2_rmagerr,k2_imag,k2_imagerr,k2_zmag,k2_zmagerr,k2_jmag,k2_jmagerr,k2_hmag,k2_hmagerr," \
            "k2_kmag,k2_kmagerr"

    # query NASA Exoplanet Archive for target data
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?" \
        "table=k2targets&select={}&where=epic_number={}".format(columns, k2id)
        
    # query API
    response = requests.get(url)
    data = [np.nan if (v == "") else float(v) for v in response.content.split("\n")[1].split(",")]
    mags, mag_errs = data[::2], data[1::2]
    
    return mags, mag_errs


def calc_teff(k2id=None, lookup=False, V=None, J=None, H=None, r=None, Verr=None, Jerr=None, Herr=None, rerr=None,
              force_rJH=False):
    # calculate effective temperature from colours according to Mann 2015, see:
    # http://adsabs.harvard.edu/abs/2015ApJ...804...64M
    # and note the erratum.

    if lookup:
        lookup = epic_lookup(k2id)
        print lookup
        kep, B, V, u, g, r, i, z, J, H, K = lookup[0]
        keperr, Berr, Verr, uerr, gerr, rerr, ierr, zerr, Jerr, Herr, Kerr = lookup[1]

    if all([V, J, H]) and not force_rJH:
        X = V - J
        Y = J - H
        Xerr = np.sqrt(Verr ** 2 + Jerr ** 2)
        Yerr = np.sqrt(Jerr ** 2 + Herr ** 2)
        a, b, c, d, e, f, g, sig = 2.769, -1.421, 0.4284, -0.06133, 0.003310, 0.1333, 0.05416, 48.
        teff = (a + b * X + c * X ** 2 + d * X ** 3 + e * X ** 4 + f * Y + g * Y ** 2) * 3500.
        teff_err_from_mag = np.sqrt((b + 2 * c * X + 3 * d * X ** 2 + 4 * e * X ** 3) ** 2 * Xerr ** 2 +
                                    (f + 2 * g * Y) ** 2 * Yerr ** 2) * 3500.
        teff_err = np.sqrt(teff_err_from_mag ** 2 + 60 ** 2 + sig ** 2)

    elif all([r, J, H]):
        X = r - J
        Y = J - H
        Xerr = np.sqrt(rerr ** 2 + Jerr ** 2)
        Yerr = np.sqrt(Jerr ** 2 + Herr ** 2)
        a, b, c, d, e, f, g, sig = 2.151, -1.092, 0.3767, -0.06292, 0.003950, 0.1697, 0.03106, 52.
        teff = (a + b * X + c * X ** 2 + d * X ** 3 + e * X ** 4 + f * Y + g * Y ** 2) * 3500.
        teff_err_from_mag = np.sqrt((b + 2 * c * X + 3 * d * X ** 2 + 4 * e * X ** 3) ** 2 * Xerr ** 2 +
                                    (f + 2 * g * Y) ** 2 * Yerr ** 2) * 3500.
        teff_err = np.sqrt(teff_err_from_mag ** 2 + 60 ** 2 + sig ** 2)

    else:
        teff = np.nan
        teff_err = np.nan

    return teff, teff_err


# testing
if __name__ == "__main__":
    print epic_lookup(247887989)

    # print(calc_teff(k2id=211799258, lookup=True, force_rJH=False))
    #
    # for epic in [231155049, 210910807, 248690431, 211799258, 211509553, 228961884]:
    #     # x = calc_teff(epic, lookup=True, force_rJH=True)
    #     kep, B, V, u, g, r, i, z, J, H, K = epic_lookup(epic)[0]
    #     print epic, V, r, K

    # MK [4, 7]
    # mk, d = 11.9, 162.
    # MK = mk - 5.*(np.log10(d) - 1.)
    # logmr = 1e-3 * (1.8 + 6.12*MK + 13.205*MK**2 - 6.2315*MK**3 + 0.37529*MK**4)
    # print MK, logmr, 10**logmr

# (nan, nan)
# (4168.891223758456, 94.12637179623471)
# (3924.3347699132833, 89.71208222309036)
# (nan, nan)
# (3660.0395368376, 89.63638201545434)
# (nan, nan)

# (nan, nan)
# (4036.476802561671, 98.49262969724671)
# (3892.715576282343, 90.77971117486133)
# (3328.5274348899966, 83.27903739338224)
# (3681.5433722782814, 86.43263312087201)
# (nan, nan)
