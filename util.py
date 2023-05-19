def calculate_k(population_size, curr_iter, N_ITER):
    """Calculate the number of individuals to be selected for tournament selection.

    Args:
        population_size (int): The size of the population.
        curr_iter (int): The current iteration number.
        N_ITER (int): The total number of iterations.

    Returns:
        int: The number of individuals to be selected for tournament selection.
    """
    return max(2, population_size * curr_iter // N_ITER)