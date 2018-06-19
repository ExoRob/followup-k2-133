import dill
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

sde_ar, per_ar = [], []

for i in range(1, 4):
    with open("bootstrap_res{}.pkl".format(i), "rb") as pf:
        cand_sde, sde_ar_tmp, per_ar_tmp = dill.load(pf)
    sde_ar = np.append(sde_ar, sde_ar_tmp)
    per_ar = np.append(per_ar, per_ar_tmp)

fps = np.sum(sde_ar > cand_sde)
print fps, "/", len(sde_ar)
print "FPP =", float(fps)/len(sde_ar)*100., "%"
print per_ar[sde_ar > cand_sde]

plt.figure(figsize=(16, 8))
plt.hist(sde_ar, bins=1000)
plt.axvline(cand_sde, c="k", ls="--")
plt.xlabel("SDE")
plt.ylabel("Number of samples")
plt.show()
