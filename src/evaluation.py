import numpy as np

from .pxp import assemble_lambda2_hamiltonian
from .dynamics import time_evolution
from .observables import return_probability
from .scoring import revival_score


def scan_lambda2(base_hamiltonian, deformation_hamiltonian, initial_index, lambda_values, time_step=0.05, max_time=30):
    scores = np.zeros(len(lambda_values))
    periods = np.full(len(lambda_values), np.nan)

    for position, lambda2 in enumerate(lambda_values):
        H = assemble_lambda2_hamiltonian(base_hamiltonian, deformation_hamiltonian, lambda2)

        times, evolved_states = time_evolution(H, initial_index, time_step=time_step, max_time=max_time)

        probabilities = return_probability(evolved_states, initial_index)

        score, period = revival_score(probabilities, time_step=times[1] - times[0])

        scores[position] = score

        if period is not None:
            periods[position] = period

    return scores, periods