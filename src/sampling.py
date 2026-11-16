import numpy as np

parameter_names = ("lambda2", "lambda3", "lambda22", "delta", "V")

parameter_bounds = np.array([
    [-0.2, 0.2],
    [-0.2, 0.2],
    [-0.2, 0.2],
    [-0.5, 0.5],
    [-0.5, 0.5]
])


def sample_parameter_vectors(number_of_samples, seed):
    rng = np.random.default_rng(seed)

    lower_bounds = parameter_bounds[:, 0]
    upper_bounds = parameter_bounds[:, 1]

    parameter_vectors = rng.uniform(low=lower_bounds, high=upper_bounds, size=(number_of_samples, len(parameter_names)))

    return parameter_vectors