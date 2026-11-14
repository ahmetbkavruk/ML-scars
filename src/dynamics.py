import numpy as np
from scipy.sparse.linalg import expm_multiply

def return_probability(H, initial_index, time_step=0.05, max_time=30):
    psi0 = np.zeros(H.shape[0], dtype=complex)
    psi0[initial_index] = 1

    number_of_steps = round(max_time / time_step)
    times = np.linspace(0, max_time, number_of_steps + 1)

    # Sparse time evolution if H is sprase
    evolved_states = expm_multiply(-1j * H, psi0, start=0, stop=max_time, num=number_of_steps + 1, endpoint=True)

    return_probability = np.abs(evolved_states[:, initial_index])**2

    return times, return_probability


def revival_score(return_probability, time_step=0.05):
    best_score = 0
    best_period = None

    minimum_period_index = round(3 / time_step)
    maximum_period_index = round(7 / time_step)

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


def norm_error(psi):
    norm_squared = np.vdot(psi, psi).real

    return np.abs(norm_squared - 1)


def energy_expectation(H, psi):
    return np.vdot(psi, H @ psi).real


def energy_error(H, psi, initial_energy):
    current_energy = energy_expectation(H, psi)

    return np.abs(current_energy - initial_energy)

def state_difference(psi1, psi2):
    return np.linalg.norm(psi1 - psi2)