import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
fig_path = Path(__file__).parent / "figures"
sys.path.insert(0, Path(__file__).parent.parent.absolute())
from src.tinygp import kernels as custom_kernels
from src.tinygp import GaussianProcess as custom_GP

import numpy as np
random = np.random.default_rng(42)


def generate_random_dataset(dset_size):
    t = np.sort(
            random.uniform(0, 3.8, dset_size),
    )
    yerr = random.uniform(0.08, 0.22, len(t))
    y = (
        0.2 * (t - 5)
        + np.sin(3 * t + 0.1 * (t - 5) ** 2)
        + yerr * random.normal(size=len(t))
    )
    return t, y, yerr

def build_custom_gp(t, yerr, params):
    kernel = np.exp(2 * params["log_sigma2"]) * custom_kernels.quasisep.Matern52(
        scale=np.exp(params["log_scale"])
    )
    return custom_GP(
        kernel,
        t,
        diag=yerr**2 + np.exp(params["log_jitter"]),
        mean=params["mean"],
    )


params = {
    "mean": 0.0,
    "log_jitter": 0.0,
    "log_sigma1": 0.0,
    "log_omega": np.log(2 * np.pi),
    "log_quality": 0.0,
    "log_sigma2": 0.0,
    "log_scale": 0.0,
}

from multiprocessing import  Pool
def apply_model(_):
    custom_model = build_custom_gp(t, yerr, params)
    custom_log_prob = -custom_model.log_probability(y)

fig, axis = plt.subplots()

orders_to_plot = [1, 5, 10, 20, 40]
cmap = plt.get_cmap("viridis")

for index, N_orders in enumerate(orders_to_plot):
    print("Starting N orders: ", N_orders)
    dset_sizes = [100, 1000, 10000, 100000]
    final_times = []
    for set_ind, dset_size in enumerate(dset_sizes):
        print("\tStarting dset size: ", dset_size)
        t, y, yerr = generate_random_dataset(dset_size)

        t_spent = time.time()
        with Pool() as p:
            p.map(apply_model, range(N_orders))
        final_time = time.time() - t_spent
        final_times.append(final_time)
        print("\t\tTook: ", final_time)

    axis.plot(dset_sizes,
             final_times,
             color=cmap(index/len(orders_to_plot)),
             label=f"N orders: {N_orders}",
             marker='x',
             ls='--'
              )

    print(final_times)
axis.set_xlabel("Dataset size")
axis.set_ylabel("Computation time [s]")
axis.legend(ncols=6, loc=4, bbox_to_anchor=(1,1))
axis.set_xscale("log")
fig.savefig(fig_path / "evol_iter_time.png", dpi=600)
plt.show()