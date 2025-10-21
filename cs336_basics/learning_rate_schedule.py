import math


def learning_rate_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, tw: int, tc: int
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear warmup) and an iteration number, return the learning rate at the given iteration under the specified schedule.

    Args:
        it: int, iteration number to get learning rate for.
        max_learning_rate: float, maximum learning rate for cosine learning rate schedule (with warmup).
        min_learning_rate: float, minimum/final learning rate for cosine learning rate schedule (with warmup).
        tw: int, number of iterations to linearly warm-up the learning rate.
        tc: int, number of cosine annealing iterations.
    """
    if it < tw:
        return max_learning_rate * it / tw
    elif it <= tc:
        return (
            min_learning_rate
            + (max_learning_rate - min_learning_rate)
            * (1 + math.cos((it - tw) / (tc - tw) * math.pi))
            / 2
        )
    else:
        return min_learning_rate
