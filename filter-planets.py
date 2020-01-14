import numpy as np
import pandas as pd
import my_constants as myc

df = pd.read_csv("all-planets.csv", skiprows=358)

mask = np.ones(len(df), bool)
mask &= (df.st_teff <= 4000.)
mask &= (df.pl_rade >= 1.) & (df.pl_rade <= 2.)

# t = df.st_teff * (1. - 0.3)**0.25 * (df.st_rad * myc.RS / 2. / df.pl_orbsmax / myc.AU)**0.5
# t = df.pl_eqt
t = df.st_teff * (1. - 0.3)**0.25 * (1. / df.pl_ratdor / 2.)**0.5
df["pl_teq"] = t

mask &= (t <= 310.) & (t >= 250.)

print 3655. * (1. - 0.3)**0.25 * (0.455 * myc.RS / 2. / 0.137 / myc.AU)**0.5

print df[["pl_hostname", "st_teff", "st_rad", "pl_orbsmax", "pl_teq"]][mask]
