import numpy as np
import my_constants as myc

G = myc.G


def t_sync(Rp, Mp, Ms, d, w_i, alpha=1./3., Qp=500., w_f=0.):
    ts = 4./9. * alpha * Qp * Rp**3. / G / Mp * (w_i - w_f) * (Mp / Ms)**2. * (d / Rp)**6.

    return ts


print t_sync(1.8*myc.RE, 10.*myc.ME, 0.461*myc.MS, 0.14*myc.AU, 1e-4) / (24. * 3600. * 365. * 1e6)
