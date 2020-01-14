import numpy as np


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
                    [0.4838350155, 0.7469822445, 0.4559837762]]).T  # rotation matrix for J2000 --> Galactic

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

    # lsr_vel = np.array([-8.5, 13.38, 6.49])
    # u = u + lsr_vel[0]
    # v = v + lsr_vel[1]
    # w = w + lsr_vel[2]

    return u, v, w      # , x, y, z


def calc_p(u, v, w, pop):
    # va = [-3.3, -8.0, -14.8, -50.0, -220.0][pop]
    # sig_u, sig_v, sig_w = [[18.3, 11.8, 7.0],
    #                        [31.4, 20.3, 13.0],
    #                        [43.1, 27.8, 17.5],
    #                        [56.1, 46.1, 35.1],
    #                        [131., 106., 85.]][pop]
    # f = [0.25, 0.42, 0.26, 0.068, 0.005][pop]

    # thin, thick, halo

    va = [-9., -48., -220.][pop]
    sig_u, sig_v, sig_w = [[43., 28., 17.],
                           [67., 51., 42.],
                           [131., 106., 85.]][pop]
    f = [0.93, 0.07, 0.006][pop]

    factor = 1. / ((2.*np.pi)**(3./2.) * sig_u * sig_v * sig_w)
    exponent = -u**2./(2. * sig_u**2.) - (v - va)**2./(2. * sig_v**2.) - w**2./(2. * sig_w**2.)

    return f * factor * np.exp(exponent)


n = 500000

# K2-133
# ra, dec = 70.149378, 25.009813
# d, du = 75.2, 0.2
# rv, rvu = 96.89, 1.67
# pmra, pmrau = 185.658, 0.082
# pmdec, pmdecu = -46.363, 0.038

# 210894022
# ra, dec = 59.889755, 21.298685
# d, du = 210., 20.
# rv, rvu = -16.3372, 0.0224
# pmra, pmrau = 122.7, 2.2
# pmdec, pmdecu = -35.3, 1.4

# 212737443 (K2-310)
# ra, dec = 204.221695, -7.318144
# d, du = 336., 5.
# rv, rvu = -16., 5.2
# pmra, pmrau = -46.2, 0.1
# pmdec, pmdecu = 22.2, 0.1

# 250143219
ra, dec = 236.1668533225586, -13.707911143006553
d, du = 186.597934467001, 1.7885680745129946
rv, rvu = -26.175, 0.062                                # TRES
# rv, rvu = -27.765875528621482, 0.6749900282084961       # Gaia
pmra, pmrau = -21.613506455511153, 0.08572600579639214
pmdec, pmdecu = -1.4833815683051608, 0.062093033995728214

# K2-260
# ra, dec = 76.867325, 16.867718
# d, du = 676., 19.
# rv, rvu = 29.1, 2.7
# pmra, pmrau = 0.667, 0.078
# pmdec, pmdecu = -6.045, 0.051

dsp = np.random.normal(d, du, n)              # Gaia distance
rvsp = np.random.normal(rv, rvu, n)           # Gaia RV
pmrasp = np.random.normal(pmra, pmrau, n)     # Gaia RA pm
pmdecsp = np.random.normal(pmdec, pmdecu, n)  # Gaia Dec pm

u, v, w = gal_uvwxyz(dsp, ra, dec, pmrasp, pmdecsp, rvsp)

u += -8.5
v += 13.38
w += 6.49

# u += 10.
# v += 5.3
# w += 7.2

vspace = np.sqrt(u**2. + v**2. + w**2.)

medians = []
for i, vel in enumerate([vspace, u, v, w]):
    v_pc = np.percentile(vel, [50., 50.-68.27/2., 50.+68.27/2.])
    vname = "S,U,V,W".split(",")[i]
    print "{} = {:6.2f} +/- {:4.2f} km/s".format(vname, v_pc[0], np.average(np.abs(v_pc[1:] - v_pc[0])))

    medians.append(v_pc[0])

s, u, v, w = medians
print

# pop_name = ["Young disk", "Interm. disk", "Old disk", "Thick disk", "Halo"]
pop_name = ["Thin disc", "Thick disc", "Halo"]
p_tot = sum(calc_p(u, v, w, pop) for pop in range(len(pop_name)))

for pop in range(len(pop_name)):
    pop_prob = calc_p(u, v, w, pop) / p_tot

    print "{:11} {:7.4f} %".format(pop_name[pop], pop_prob*100.)
