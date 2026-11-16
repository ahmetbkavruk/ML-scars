import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize_scalar

def time_evolution(H, initial_index, time_step=0.05, max_time=30):
    psi0 = np.zeros(H.shape[0], dtype=complex)
    psi0[initial_index] = 1

    number_of_steps = int(np.ceil(max_time / time_step))
    times = np.linspace(0, max_time, number_of_steps + 1)

    evolved_states = expm_multiply(-1j * H, psi0, start=0, stop=max_time, num=number_of_steps + 1, endpoint=True)

    return times, evolved_states