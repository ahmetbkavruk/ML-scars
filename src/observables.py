import numpy as np
from .dynamics import time_evolution

def return_probability(H, initial_index, time_step=0.05, max_time=30):
    times, evolved_states = time_evolution(H, initial_index, time_step, max_time)

    probabilities = np.abs(evolved_states[:, initial_index])**2

    return times, probabilities


def revival_score(return_probability, time_step=0.05):
    best_score = 0
    best_period = None

    minimum_period_index = int(np.ceil(3 / time_step))
    maximum_period_index = int(np.floor(7 / time_step))

    for period_index in range(minimum_period_index, maximum_period_index + 1):
        # Departure condition
        minimum_probability = np.min(return_probability[:period_index + 1])

        if minimum_probability > 0.2:
            continue

        first_revival = return_probability[period_index]
        second_revival = return_probability[2 * period_index]
        third_revival = return_probability[3 * period_index]

        score = (first_revival + second_revival + third_revival) / 3

        if score > best_score:
            best_score = score
            best_period = period_index * time_step

    return best_score, best_period


def state_infidelity(psi1, psi2):
    overlap = np.vdot(psi1, psi2)

    norm1 = np.vdot(psi1, psi1).real
    norm2 = np.vdot(psi2, psi2).real

    fidelity = abs(overlap)**2 / (norm1 * norm2)

    return max(0.0, 1 - fidelity)


def energy_expectation(H, psi):
    return np.vdot(psi, H @ psi).real


def staggered_density(H, initial_index, state, indices, time_step=0.05, max_time=30):
    times, evolved_states = time_evolution(H, initial_index, time_step, max_time)

    staggered_values = np.zeros(len(indices))

    for position, i in enumerate(indices):

        for site in range(state.L):
            n_i = (1 - state.z(i, site)) / 2
            sign = (-1)**site

            staggered_values[position] += sign * n_i / state.L

    probabilities = np.abs(evolved_states)**2

    densities = probabilities @ staggered_values

    return times, densities


def entanglement_entropy(H, initial_index, state, indices, time_step=0.05, max_time=30):
    times, evolved_states = time_evolution(H, initial_index, time_step, max_time)

    left_size = state.L // 2
    right_size = state.L - left_size

    left_dimension = 2**left_size
    right_dimension = 2**right_size

    right_mask = 2**right_size - 1

    left_indices = []
    right_indices = []

    for index in indices:

        left_index = index >> right_size
        right_index = index & right_mask

        left_indices.append(left_index)
        right_indices.append(right_index)

    entropies = []

    for psi in evolved_states:
        coefficient_matrix = np.zeros((left_dimension, right_dimension), dtype=complex)

        for position in range(len(indices)):
            left_index = left_indices[position]
            right_index = right_indices[position]

            coefficient_matrix[left_index, right_index] = psi[position]

        # Schmidt coefficients
        singular_values = np.linalg.svd(coefficient_matrix, compute_uv=False)

        eigenvalues = singular_values**2

        # Avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-14]

        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        entropies.append(entropy)

    return times, np.array(entropies)