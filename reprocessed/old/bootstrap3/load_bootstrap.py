import dill
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

sde_ar, per_ar = [], []

for i in range(0, 3):
    with open("bootstrap_res{}.pkl".format(i), "rb") as pf:
        cand_sde, sde_ar_tmp, per_ar_tmp = dill.load(pf)
    sde_ar = np.append(sde_ar, sde_ar_tmp)
    per_ar = np.append(per_ar, per_ar_tmp)

# with open("from-kelvin/bootstrap_res1.pkl", "rb") as pf:
#     cand_sde, sde_ar, per_ar = dill.load(pf)

fps = np.sum(sde_ar > cand_sde)
print fps, "/", len(sde_ar)
print "FPP =", float(fps)/len(sde_ar)*100., "%"

pers = per_ar[sde_ar > cand_sde]
# print pers

sns.distplot(pers, kde=False, bins=20)
plt.axvline(26.6, c="k", ls="--")
plt.xlim(0)
plt.xlabel("Period (d)")
plt.show()

# plt.figure(figsize=(16, 8))
# plt.hist(sde_ar, bins=500)
# plt.axvline(cand_sde, c="k", ls="--")
# plt.xlabel("SDE")
# plt.ylabel("Number of samples")
# plt.show()

plt.figure(figsize=(14, 7))
sns.distplot(sde_ar, bins=100, kde=True, hist=True, label="Bootstrap")
plt.axvline(cand_sde, color="k", linestyle="--", label="Candidate")
plt.xlabel("SDE")
plt.legend()
plt.tight_layout()
plt.show()
