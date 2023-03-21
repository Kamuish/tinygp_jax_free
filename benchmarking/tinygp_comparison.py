import time
import sys
import time
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.parent.absolute())
import numpy as np
import matplotlib.pyplot as plt
from src.tinygp import kernels as custom_kernels
from src.tinygp import GaussianProcess as custom_GP
import jax
jax.config.update("jax_enable_x64", True)
from tinygp import kernels, GaussianProcess

random = np.random.default_rng(42)

def generate_random_dataset():
    t = np.sort(
        np.append(
            random.uniform(0, 3.8, 1008),
            random.uniform(5.5, 10, 18),
        )
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
def build_gp(t, yerr, params):
    kernel = np.exp(2 * params["log_sigma2"]) * kernels.quasisep.Matern52(
        scale=np.exp(params["log_scale"])
    )
    return GaussianProcess(
        kernel,
        t,
        diag=yerr**2 + np.exp(params["log_jitter"]),
        mean=params["mean"],
    )

def loss(t, y, yerr, params):

    t0 = time.time()
    gp = build_gp(t, yerr, params)
    log_prob = -gp.log_probability(y)
    print("tinyhp", time.time() - t0)

    t0 = time.time()
    custom_model = build_custom_gp(t, yerr, params)
    custom_log_prob = -custom_model.log_probability(y)
    print("Custom", time.time() - t0)

    # print(gp.solver)
    print(log_prob, custom_log_prob)
    return np.allclose(log_prob, custom_log_prob)


params = {
    "mean": 0.0,
    "log_jitter": 0.0,
    "log_sigma1": 0.0,
    "log_omega": np.log(2 * np.pi),
    "log_quality": 0.0,
    "log_sigma2": 1.0,
    "log_scale": 2.0,
}

t, y, yerr = generate_random_dataset()
# for _
for _ in range(2):
    print("Both are equal: ", loss(t, y, yerr, params))

