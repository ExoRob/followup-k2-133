import dill
import matplotlib.pyplot as plt
import numpy as np

with open("test_knots.pkl", "rb") as pf:
    n_samples, time_knots_sample, rot_knots_sample, arc_knots_sample, res_ar, ph_res_ar = dill.load(pf)

# mask = np.isfinite(res_ar)

ind = np.nanargmin(res_ar)
# ind = np.nanargmin(ph_res_ar)

print res_ar[ind], ph_res_ar[ind], time_knots_sample[ind], rot_knots_sample[ind], arc_knots_sample[ind]

plt.plot(ph_res_ar, "o")
plt.plot(res_ar, "o")
plt.show()

# plt.plot(time_knots_sample, ph_res_ar, ".")
# plt.plot(rot_knots_sample, ph_res_ar, ".")
# plt.plot(arc_knots_sample, ph_res_ar, ".")
# plt.show()
