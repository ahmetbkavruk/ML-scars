import numpy as np
from scipy.sparse.linalg import expm_multiply

from .dynamics import time_evolution
from .observables import (return_probability, 
    state_infidelity,
    energy_expectation,
)
from .scoring import revival_score


def norm_error(psi):
    norm_squared = np.vdot(psi, psi).real

    return np.abs(norm_squared - 1)


def energy_error(H, psi, initial_energy):
    current_energy = energy_expectation(H, psi)

    return np.abs(current_energy - initial_energy)


def state_difference(psi1, psi2):
    return np.linalg.norm(psi1 - psi2)


def benchmark_time_evolution(H, psi0, max_time=30, time_step=0.05):
    number_of_steps = int(np.ceil(max_time / time_step))
    times = np.linspace(0, max_time, number_of_steps + 1)

    # Sparse time evolution
    sparse_states = expm_multiply(-1j * H, psi0, start=0, stop=max_time, num=number_of_steps + 1, endpoint=True)

    # Dense diagonalization
    eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
    coefficients = eigenvectors.conj().T @ psi0

    initial_energy = energy_expectation(H, psi0)

    state_differences = []
    infidelities = []
    norm_errors = []
    energy_errors = []

    for time_index, time in enumerate(times):
        sparse_state = sparse_states[time_index]

        phases = np.exp(-1j * eigenvalues * time)
        dense_state = eigenvectors @ (phases * coefficients)

        state_differences.append(state_difference(sparse_state, dense_state))
        
        infidelities.append(state_infidelity(sparse_state, dense_state))

        norm_errors.append(norm_error(sparse_state))

        energy_errors.append(energy_error(H, sparse_state, initial_energy))

    return {
        "times": times,
        "state_differences": np.array(state_differences),
        "infidelities": np.array(infidelities),
        "norm_errors": np.array(norm_errors),
        "energy_errors": np.array(energy_errors)
    }



def benchmark_revivals(H, initial_index, initial_time_step=0.05, refinements=4, max_time=30, tolerance=1e-3):
    previous_score = None
    previous_period = None

    results = []

    for refinement in range(refinements):
        time_step = initial_time_step / (2**refinement)

        times, evolved_states = time_evolution(H, initial_index, time_step=time_step, max_time=max_time)

        probabilities = return_probability(evolved_states, initial_index)

        score, period = revival_score(probabilities, time_step=times[1] - times[0])

        score_difference = None
        period_difference = None

        if previous_score is not None:
            score_difference = abs(score - previous_score)

            if period is not None and previous_period is not None:
                period_difference = abs(period - previous_period)

        print(f"Time step: {times[1] - times[0]:.8f}")
        print(f"Revival score: {score:.8f}")
        print(f"Optimal period: {period}")

        if score_difference is not None:
            print(f"Score difference: {score_difference:.8e}")

            if period_difference is not None:
                print(f"Period difference: {period_difference:.8e}")

            if score_difference < tolerance:
                print("Score convergence criterion satisfied.")

        print()

        results.append({
            "time_step": times[1] - times[0],
            "score": score,
            "period": period,
            "score_difference": score_difference,
            "period_difference": period_difference
        })

        previous_score = score
        previous_period = period

    return results
