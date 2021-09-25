import pymc3 as pm
import numpy as np
from pymc3 import Model as pymc3_model
from pymc3.backends.base import MultiTrace
from pymc3 import *
import matplotlib.pyplot as plt


def prob_model() -> pymc3_model:
    with pm.Model() as regr_model:
        p_a = pm.Normal('p_a' , 0, 2)
        p_b = pm.Normal('p_b', 1, 1)

        mu = regression_model(p_a, p_b, x(200))

        likelihood = pm.Normal("p_y", mu=mu, sigma=1, observed=observed(200))

    return regr_model


def x(size: int):
    return np.linspace(0, 1, size)


def observed(size: int):
    return regression_model(0.8, 1.2, x(size)) + np.random.normal(scale=0.5, size=size)


def regression_model(a, b, x):
    return a + b*x


def sample():
    size = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + np.random.normal(scale=0.5, size=size)
    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10, testval=1.0)
        intercept = Normal("Intercept", 0, sigma=20)
        x_coeff = Normal("x", 0, sigma=20)

        # Define likelihood
        likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)

        # Inference!
        trace = pm.sample(3000, cores=2)

    #with pm.Model() as regr_model:
    #    p_a = pm.Normal('p_a' , 0, 2)
    #    p_b = pm.Normal('p_b', 1, 1)

    #    mu = regression_model(p_a, p_b, x(200))

    #    likelihood = pm.Normal("p_y", mu=mu, sigma=1, observed=observed(200))

        #step = pm.NUTS(target_accept=0.9)
    #    trace = pm.sample(draws=100000, chains=2, discard_tuned_samples=True, progressbar=True, cores=1)

    #with prob_model() as model:
    #    step = pm.NUTS(target_accept=0.9)
    #    trace = pm.sample(draws=100000, chains=2, discard_tuned_samples=True, progressbar=True, cores=1)

    return trace


def plot_trace(trace: MultiTrace):
    plt.figure(figsize=(7, 7))
    traceplot(trace)
    plt.tight_layout()
