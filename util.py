def calculate_k(population_size, curr_iter, N_ITER):
    return max(2, population_size * curr_iter // N_ITER)