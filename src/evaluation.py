import numpy as np
import time
import pandas as pd

from .pxp import assemble_lambda2_hamiltonian, build_deformed_pxp_hamiltonian
from .dynamics import time_evolution
from .observables import return_probability
from .scoring import revival_score
from .sampling import parameter_names


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


def evaluate_parameter_vector(parameters, state, indices, initial_index, time_step=0.05, max_time=30):
    parameters = np.asarray(parameters, dtype=float)

    start_time = time.perf_counter()

    full_hamiltonian = build_deformed_pxp_hamiltonian(state, **dict(zip(parameter_names, parameters)))

    hamiltonian = full_hamiltonian[indices, :][:, indices].tocsr()

    times, evolved_states = time_evolution(hamiltonian,initial_index,time_step=time_step,max_time=max_time)

    probabilities = return_probability(evolved_states, initial_index)

    score, period = revival_score(probabilities,time_step=time_step)

    runtime = time.perf_counter() - start_time

    result = dict(zip(parameter_names, parameters))

    result["score"] = float(score)
    result["period"] = np.nan if period is None else float(period)
    result["runtime_seconds"] = runtime
    result["L"] = state.L
    result["time_step"] = time_step
    result["max_time"] = max_time

    return result


def evaluate_parameter_vectors(parameter_vectors, state, indices, initial_index, time_step=0.05, max_time=30):
    results = []

    for parameters in parameter_vectors:
        result = evaluate_parameter_vector(parameters, state, indices, initial_index, time_step=time_step, max_time=max_time)

        results.append(result)

    return pd.DataFrame(results)