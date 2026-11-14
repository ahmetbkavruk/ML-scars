import numpy as np
from scipy.sparse.linalg import expm_multiply

from .dynamics import state_difference, state_infidelity, norm_error, energy_expectation, energy_error

def benchmark_time_evolution(H, psi0, max_time=30, time_step=0.05):
    number_of_steps = round(max_time / time_step)
    times = np.linspace(0, max_time, number_of_steps + 1)

    # Sparse time evolution
    sparse_states = expm_multiply(
        -1j * H,
        psi0,
        start=0,
        stop=max_time,
        num=number_of_steps + 1,
        endpoint=True
    )

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
