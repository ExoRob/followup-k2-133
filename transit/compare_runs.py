import dill
import numpy as np
import pandas
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
sns.set()

runs = ["save_K2SC_mask_16_300_2000_1000", "save_K2SFF_16_300_2000_1000", "save_LKSFF_16_300_2000_1000"]
n = len(runs)
all_df = pandas.read_csv(runs[0] + "/fit_vals.csv")     # combine to one dataframe
for run in runs[1:]:
    df = pandas.read_csv(run + "/fit_vals.csv").drop('Property', 1)
    all_df = pandas.concat([all_df, df], axis=1, sort=False)

samps = []
for run in runs:    # make list of all samples
    pklfile = run + "/mcmc.pkl"
    with open(pklfile, "rb") as pklf:
        data, planet, samples = dill.load(pklf)
    samps.append(samples)

columns = ["Property"]
methods = []
for run in runs:
    method = run.split("_")[1]
    methods.append(method)
    columns += [s + "_" + method for s in [u'b', u'c', u'd', u'01']]
all_df.columns = columns

# print all_df.columns
# print all_df.Property

# plot gaussian distributions planets x params x runs
pbar = tqdm.tqdm(total=16, initial=0, desc="Making KDE plots")
for planet in range(4):                             # each planet (new figure)
    pl = ["d", "c", "b", "01"][planet]
    fig = plt.figure(figsize=(10, 8))
    for p in range(planet*4, planet*4+4):           # each parameter of fit (new subplot)
        ax = fig.add_subplot(2, 2, (p % 4) + 1)
        for r in range(n):                          # each run (new distribution)
            yvals = samps[r].T[p]
            ax = sns.kdeplot(yvals, shade=True, label=(methods[r] if p % 4 == 1 else ""))
            ax.tick_params(axis='y', labelleft=False)
        ax.set_title(["Rp/Rs", "Inclination", "Epoch", "Period"][p % 4], x=0.2, y=0.8)
        pbar.update(1)

    plt.legend()
    fig.tight_layout()
    plt.suptitle("Planet {}".format(pl), fontweight="bold")
    plt.savefig("planet_{}_kde.pdf".format(pl))
    plt.close("all")
    # plt.show()
pbar.close()


# # violin plot planets x params x runs
# pbar = tqdm.tqdm(total=16, initial=0, desc="Making violin plots")
# for planet in range(4):                             # each planet (new figure)
#     pl = ["d", "c", "b", "01"][planet]
#     fig = plt.figure(figsize=(10, 8))
#     for p in range(planet*4, planet*4+4):               # each parameter of fit (new subplot)
#         x = []
#         y = []
#         for r in range(n):                          # each run (new violin)
#             yvals = samps[r].T[p]
#             x += [methods[r] for i in range(len(yvals))]
#             y += list(yvals)
#
#         ax = fig.add_subplot(2, 2, (p % 4) + 1)
#         ax = sns.violinplot(x=x, y=y)
#         ax.set_title(["Rp/Rs", "Inclination", "Epoch", "Period"][p % 4])
#         pbar.update(1)
#
#     plt.suptitle("Planet {}".format(pl))
#     fig.tight_layout()
#     plt.savefig("planet_{}_violin.pdf".format(pl))
#     plt.close("all")
#     # plt.show()
