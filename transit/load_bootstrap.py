import dill
import matplotlib.pyplot as plt


with open("bootstrap_res.pkl", "rb") as pf:
    cand_sde, sde_ar, per_ar = dill.load(pf)

plt.hist(sde_ar, bins=10)
plt.show()
