import numpy as np

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