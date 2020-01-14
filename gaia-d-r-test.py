# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import my_constants as myc
import h5py
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm
import corner
from mk_mass import posterior
import seaborn as sns
sns.set()

# d = 75.2 * myc.PC
# teff = 3630.
# L = 10.**-1.47
# md = 1.9253953263879E-10

# L / LS = (R / RS)^2 (T / TS)^4
# R / RS = ((L / LS) (TS / T)^4)^0.5

# print((L * (5777. / teff)**4.)**0.5)
#
# print (md * d / myc.RS)**2.


# --------


bcmodel = h5py.File('bcgrid.h5', 'r')
interp = RegularGridInterpolator((np.array(bcmodel['teffgrid']),
                                  np.array(bcmodel['logggrid']), np.array(bcmodel['fehgrid']),
                                  np.array(bcmodel['avgrid'])), np.array(bcmodel['bc_k']))


def getbc(teff, logg, feh, av, method="huber", mags=None, z=None):
    """
    Compute BC-K from model grid or Mann relations
    :param teff: Teff (K)
    :param logg: log g
    :param feh: metallicity
    :param av: extinction
    :param method: huber, mann-vj or mann-rj
    :param mags: magnitudes needed for Mann relations - (V,J) or (r,J)
    :return: Bolometric correction in K-band
    """
    if "mann" in method:
        assert (len(mags) == 2), "The Mann relation requires 2 mags."
        x = mags[0] - mags[1]

    if method == "huber":
        return interp(np.array([teff, logg, feh, av]))[0]

    elif method == "mann-vj":
        return np.poly1d([1.421, 0.6084, -0.09655, 6.263e-3][::-1])(x)

    elif method == "mann-rj":
        if z is None:
            return np.poly1d([1.719, 0.5236, -0.09085, 6.735e-3][::-1])(x)
        else:
            return np.poly1d([1.572, 0.6529, -0.1260, 9.746e-3][::-1])(x) + 0.08987*z

    else:
        raise NotImplementedError


def calc_R(d, teff, bc, A, mk):
    """
    Calculate stellar radius from bolometric luminosity
    :param d: distance (pc)
    :param teff: Teff (K)
    :param bc: bolometric correction
    :param A: extinction
    :param mk: K mag
    :return: Radius
    """
    # https://arxiv.org/pdf/1805.01453.pdf
    # http://argonaut.skymaps.info/query
    # Lbol = L0 * 10^(0.4 * Mbol)
    # L0 = 3.0128 * 10^28 W
    # Mbol = m - A - mu - BC    # any apparent m, extinction, distance modulus, correction

    mu = 5.*np.log10(d / 10.)   # distance modulus

    Mbol = mk - A - mu + bc     # absolute bolometric mag
    Lbol = 3.0128e28 * 10.**(-0.4 * Mbol)    # bolometric luminosity

    R = (Lbol / (4. * np.pi * 5.67e-8 * teff**4.))**0.5 / myc.RS  # stellar radius

    return R, Lbol / 3.828e26


def gal_uvwxyz(d, ra, dec, pmra, pmdec, rv):
    """
    Converted from https://idlastro.gsfc.nasa.gov/ftp/pro/astro/gal_uvw.pro
    :param d: distance, pc
    :param ra: RA, deg.
    :param dec: Dec, deg.
    :param pmra: RA proper motion, mas/yr
    :param pmdec: Dec proper motion, mas/yr
    :param rv: radial velocity, km/s
    :return: U,V,W
    """
    cosd = np.cos(dec * np.pi / 180.)
    sind = np.sin(dec * np.pi / 180.)
    cosa = np.cos(ra * np.pi / 180.)
    sina = np.sin(ra * np.pi / 180.)

    k = 4.74047  # Equivalent of 1 A.U/yr in km/s
    a_g = np.array([[0.0548755604, 0.4941094279, -0.8676661490],
                    [0.8734370902, -0.4448296300, -0.1980763734],
                    [0.4838350155, 0.7469822445, 0.4559837762]]).T  # rotation matrix for J2000 -->Galactic

    # AR 2013.0910: In order to use this with vectors, we need more control over the matrix multiplication
    # pos1 = cosd * cosa
    # pos2 = cosd * sina
    # pos3 = sind
    # x = d * (a_g[0, 0] * pos1 + a_g[1, 0] * pos2 + a_g[2, 0] * pos3)
    # y = d * (a_g[0, 1] * pos1 + a_g[1, 1] * pos2 + a_g[2, 1] * pos3)
    # z = d * (a_g[0, 2] * pos1 + a_g[1, 2] * pos2 + a_g[2, 2] * pos3)

    vec1 = rv
    vec2 = k * pmra * d / 1e3
    vec3 = k * pmdec * d / 1e3

    u = (a_g[0, 0] * cosa * cosd + a_g[0, 1] * sina * cosd + a_g[0, 2] * sind) * vec1 + \
        (-a_g[0, 0] * sina + a_g[0, 1] * cosa) * vec2 + \
        (-a_g[0, 0] * cosa * sind - a_g[0, 1] * sina * sind + a_g[0, 2] * cosd) * vec3

    v = (a_g[1, 0] * cosa * cosd + a_g[1, 1] * sina * cosd + a_g[1, 2] * sind) * vec1 + \
        (-a_g[1, 0] * sina + a_g[1, 1] * cosa) * vec2 + \
        (-a_g[1, 0] * cosa * sind - a_g[1, 1] * sina * sind + a_g[1, 2] * cosd) * vec3

    w = (a_g[2, 0] * cosa * cosd + a_g[2, 1] * sina * cosd + a_g[2, 2] * sind) * vec1 + \
        (-a_g[2, 0] * sina + a_g[2, 1] * cosa) * vec2 + \
        (-a_g[2, 0] * cosa * sind - a_g[2, 1] * sina * sind + a_g[2, 2] * cosd) * vec3

    lsr_vel = np.array([-8.5, 13.38, 6.49])
    u = u + lsr_vel[0]
    v = v + lsr_vel[1]
    w = w + lsr_vel[2]

    return u, v, w      # , x, y, z


# metal-poor values
# z = -1.
z = None

n = 400000
lg = 5.
confs = [50., 50.-68.2689/2., 50.+68.2689/2.]

d, du = 75.2, 0.2

if z is None:
    teff, teffu = 3655., 80.
elif z == -1.:
    teff, teffu = 3500., 80.

mk, mku = 10.279, 0.018
mr, mru = 13.451, 0.043
mj, mju = 11.084, 0.021

rv, rvu = 96.89, 1.67
pmra, pmrau = 185.658, 0.082
pmdec, pmdecu = -46.363, 0.038

dsp = np.random.normal(d, du, n)             # Gaia distance
teffsp = np.random.normal(teff, teffu, n)    # Mann / spectroscopic Teff
mrsp = np.random.normal(mr, mru, n)          # SDSS r mag
mjsp = np.random.normal(mj, mju, n)          # 2MASS J mag
mksp = np.random.normal(mk, mku, n)          # 2MASS K mag
asp = np.random.normal(0., 0., n)            # Bayestar17 extinction - basically zero nearby
msc = np.random.uniform(0., 0., n)           # feh

rvsp = np.random.normal(rv, rvu, n)           # Gaia RV
pmrasp = np.random.normal(pmra, pmrau, n)     # Gaia RA pm
pmdecsp = np.random.normal(pmdec, pmdecu, n)  # Gaia Dec pm


u, v, w = gal_uvwxyz(dsp, 70.149378, 25.009813, pmrasp, pmdecsp, rvsp)
vspace = np.sqrt(u**2. + v**2. + w**2.)

for i, vel in enumerate([vspace, u, v, w]):
    v_pc = np.percentile(vel, confs)
    vname = "S,U,V,W".split(",")[i]
    print "{} = {:6.2f} +/- {:4.2f} km/s".format(vname, v_pc[0], np.average(np.abs(v_pc[1:] - v_pc[0])))

# sns.distplot(vspace, bins=100)
# plt.show()


# bcsp_h = np.ones(n)
# bcsp_m = np.ones(n)
# for i in tqdm(range(n)):
#     bcsp_h[i] = getbc(teffsp[i], lg, msc[i], asp[i], "huber")  # bolometric correction
#     bcsp_m[i] = getbc(teffsp[i], lg, msc[i], asp[i], "mann-rj", (mrsp[i], mjsp[i])) + np.random.normal(0., 0.036)
#
# print bcsp_h.mean(), bcsp_m.mean()
# plt.hist(bcsp_h, bins=100, alpha=0.7)
# plt.hist(bcsp_m, bins=100, alpha=0.7)
# plt.show()

if z is None:
    bcsp = getbc(teffsp, lg, msc, asp, "mann-rj", (mrsp, mjsp)) + np.random.normal(0., 0.036, n)    # sigma
else:
    bcsp = getbc(teffsp, lg, msc, asp, "mann-rj", (mrsp, mjsp), z=z) + np.random.normal(0., 0.03, n)    # sigma

rsp, lsp = calc_R(dsp, teffsp, bcsp, asp, mksp)

r_pc = np.percentile(rsp, confs)
print "\nRs = {:.4f} $\pm$ {:.4f} Rsun".format(r_pc[0], np.average(np.abs(r_pc[1:] - r_pc[0])))

MK = mksp - 5.*(np.log10(dsp) - 1.)

# msp = np.poly1d([0.5858, 0.3872, -0.1217, 0.0106, -2.7262e-4][::-1])(MK)      # original Mann paper
# msp += np.random.normal(0., 0.018, n) * msp

if z is None:
    msp = posterior(mk, d, mku, du, feh=None, efeh=None)     # https://github.com/awmann/M_-M_K-
else:
    msp = posterior(mk, d, mku, du, feh=z, efeh=0.15)     # https://github.com/awmann/M_-M_K-

MK = np.median(MK)
m_sc = np.percentile(msp, confs)
l_sc = np.percentile(lsp, confs)

print "Ms = {:.4f} $\pm$ {:.4f} Msun\nLs = {:.4f} $\pm$ {:.4f} Lsun".\
    format(m_sc[0], np.average(np.abs(m_sc[1:] - m_sc[0])), l_sc[0], np.average(np.abs(l_sc[1:] - l_sc[0])))

# fig = corner.corner(np.array([rsp, msp, dsp, teffsp, bcsp, mksp]).T, quantiles=np.asarray(confs)/100.,
#                     show_titles=True, labels=[r"$R$", r"$M$", r"$d$", r"$T_{eff}$", r"$BC$", r"$K$"])
fig = corner.corner(np.array([rsp, msp, lsp]).T, quantiles=np.asarray(confs)/100., bins=50,
                    show_titles=True, labels=[r"$R$", r"$M$", r"$L$"], title_fmt=".3f")
plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(12, 3))
# for i, sp in enumerate([rsp, msp]):
#     sns.distplot(sp, ax=axes[i], bins=100)
#     _ = [axes[i].axvline(q, color="0.7") for q in np.percentile(sp, confs)]
#     axes[i].set_xlabel(["Radius", "Mass"][i])
# plt.show()

# # md = (R/d)^2
# md = 1.932e-21
# d *= myc.PC
# L = 1.710e-1
#
# print (md * d**2.)**0.5 / myc.RS
# print (L * (5777. / teff)**4.)**0.5


# import astropy.io.ascii as ascii
# from scipy import interpolate
# mam = ascii.read('/Users/rwells/Downloads/mamajek_asciiread2.txt', fill_values=[('....', '0.0'), ('...', '0.0')])
# Tgrid = mam['Teff']
# Mgrid = mam['M_Ks']
# fTM = interpolate.interp1d(Mgrid, Tgrid)
#
# print "Mamajek Teff = {:.0f} K".format(fTM(MK))
