import numpy as np
import pandas
import matplotlib.pyplot as plt


# with open("table1.txt", "r") as f:
#     lines = f.readlines()
#
# for line in lines[:30]:
#     print line

s = """  1         ---       Planet name
  2         ---       Planetary mass [Earth masses]
  3         ---       Planetary mass +/-34.1% uncertainty [Earth masses]
  4         ---       Planetary radius  2.2% quantile [Earth radii]
  5         ---       Planetary radius 15.9% quantile [Earth radii]
  6         ---       Planetary radius 50.0% quantile [Earth radii]
  7         ---       Planetary radius 84.1% quantile [Earth radii]
  8         ---       Planetary radius 97.7% quantile [Earth radii]
  9         ---       Stellar radius [Solar radii]
  10         ---      Stellar radius -34.1% uncertainty [Solar radii]
  11         ---      Stellar radius +34.1% uncertainty [Solar radii]
  12        ---       Transit depth  2.2% quantile [parts per thousand]
  13        ---       Transit depth 15.9% quantile [parts per thousand]
  14        ---       Transit depth 50.0% quantile [parts per thousand]
  15        ---       Transit depth 84.1% quantile [parts per thousand]
  16        ---       Transit depth 97.7% quantile [parts per thousand]
  17        ---       V-band apparent magnitude
  18        ---       CHEOPS r.m.s. photometric noise [parts per million/hour]
  19        ---       CHEOPS signal-to-noise ratio per hour 2.2% quantile
  20        ---       CHEOPS signal-to-noise ratio per hour 15.9% quantile
  21        ---       CHEOPS signal-to-noise ratio per hour 50.0% quantile
  22        ---       CHEOPS signal-to-noise ratio per hour 84.1% quantile
  23        ---       CHEOPS signal-to-noise ratio per hour 97.7% quantile
  24        ---       Maximum transit duration [hours]
  25        ---       Maximum transit duration -34.1% uncertainty [hours]
  26        ---       Maximum transit duration +34.1% uncertainty [hours]
  27        ---       Geometric transit probability [%]
  28        ---       Planet flag""".split("\n")
col_names = []
for i in range(len(s)):
    c = s[i].split("---")[-1].lstrip()
    col_names.append(c)
# print col_names

df = pandas.read_csv("table1.txt", delimiter="\t", skiprows=53)
df.columns = col_names

st_r = df[col_names[8]]
v_mag = df[col_names[16]]
ppm_hr = df[col_names[17]]

mask = (ppm_hr < 1e3)
# mask &= (st_r < 0.8) & (st_r > 0.3)


plt.plot(v_mag[mask], ppm_hr[mask], ".")
# plt.plot(st_r[mask], ppm_hr[mask], ".")
# plt.plot(st_r[mask], v_mag[mask], ".")
# plt.xlim(0., 1.)
plt.show()
