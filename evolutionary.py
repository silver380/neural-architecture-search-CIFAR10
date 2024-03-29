from chromosome import Chromosome
import random
import util


class EvolutionaryAlgorithm:
    def __init__(self, n_iter, mut_prob, recomb_prob, population_size, epochs, num_test, dataloaders, dataset_sizes):
        """
                Initialize the EvolutionaryAlgorithm class.

                Args:
                    n_iter (int): Number of iterations.
                    mut_prob (float): Probability of mutation.
                    recomb_prob (float): Probability of recombination.
                    population_size (int): Size of the population.
                    epochs (int): Number of epochs for training.
                    num_test (int): Number of tests for fitness calculation.
                    dataloaders (dict): Dictionary of data loaders for training and validation datasets.
                    dataset_sizes (dict): Dictionary of dataset sizes for training and validation datasets.
                """
        self.n_iter = n_iter
        self.mut_prob = mut_prob
        self.recomb_prob = recomb_prob
        self.epochs = epochs
        self.num_test = num_test
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.population = []
        self.population_size = population_size
        self.current_iter = 0
        self.fitness_avg = 0
        self.fitness_history = []

    # Random initialization
    def init_population(self):
        """Initialize the population with random chromosomes."""
        for _ in range(self.population_size):
            young_pop = Chromosome(self.mut_prob, self.recomb_prob, self.epochs, self.num_test, self.dataloaders, self.dataset_sizes)
            self.population.append(young_pop)

    # Fitness Tournament selection
    def tournament_selection(self, tour_pop, k):
        """
        Perform fitness tournament selection.

        Args:
            tour_pop (list): List of chromosomes to select from.
            k (int): Number of participants in the tournament.

        Returns:
            Chromosome: Selected chromosome.
        """
        parents = random.sample(tour_pop, k=k)
        parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
        bestparent = parents[0]

        return bestparent

    def parent_selection(self):
        """
        Perform parent selection using tournament selection.

        Returns:
            list: List of selected parent chromosomes.
        """
        parents = []
        for _ in range(self.population_size):
            best_parent = self.tournament_selection(self.population,
                                                    util.calculate_k(len(self.population), self.current_iter, self.n_iter))
            parents.append(best_parent)

        return parents

    # One-point crossover
    def recombination(self, mating_pool):
        """
        Perform recombination (crossover) between chromosomes.

        Args:
            mating_pool (list): List of parent chromosomes.

        Returns:
            list: List of offspring chromosomes.
        """
        youngs = []
        for _ in range(self.population_size // 2):
            parents = random.choices(mating_pool, k=2)
            young_1 = Chromosome(self.mut_prob, self.recomb_prob, self.epochs, self.num_test, self.dataloaders, self.dataset_sizes)
            young_2 = Chromosome(self.mut_prob, self.recomb_prob, self.epochs, self.num_test, self.dataloaders, self.dataset_sizes)
            prob = random.uniform(0, 1)
            if prob <= self.recomb_prob and min(len(parents[0].net['mlp']),len(parents[1].net['mlp'])) > 0:
                crossover_point = crossover_point = random.randint(0, max(min(len(parents[0].net['mlp']),
                                                                              len(parents[1].net['mlp'])) - 1, 0))
                young_1.net['mlp'] = parents[0].net['mlp'][:crossover_point].copy() + parents[1].net['mlp'][
                                                                                      crossover_point:].copy()
                young_2.net['mlp'] = parents[1].net['mlp'][:crossover_point].copy() + parents[0].net['mlp'][
                                                                                      crossover_point:].copy()
            else:
                young_1.net['mlp'] = parents[0].net['mlp'].copy()
                young_2.net['mlp'] = parents[1].net['mlp'].copy()

            youngs.append(young_1)
            youngs.append(young_2)

        return youngs

    def survival_selection(self, youngs):
        """
        Perform survival selection to determine the next population.

        Args:
            youngs (list): List of offspring chromosomes.

        Returns:
            list: List of chromosomes for the next population.
        """
        mpl = self.population.copy() + youngs
        mpl = sorted(mpl, key=lambda agent: agent.fitness, reverse=True)
        mpl = mpl[:self.population_size].copy()

        return mpl

    def mutation(self, youngs):
        """
        Perform mutation on the offspring chromosomes.

        Args:
            youngs (list): List of offspring chromosomes.

        Returns:
            list: List of mutated chromosomes.
        """
        for young in youngs:
            young.mutation()

        return youngs

    def calculate_fitness_avg(self):
        """Calculate the average fitness of the current population. """
        self.fitness_avg = 0
        for pop in self.population:
            self.fitness_avg += pop.fitness

        self.fitness_avg /= self.population_size

    def run(self):
        """
        Run the evolutionary algorithm.

        Returns:
            tuple: A tuple containing the best chromosome and the list of average fitness values over iterations.
        """
        self.init_population()

        for _ in range(self.n_iter):
            parents = self.parent_selection().copy()
            youngs = self.recombination(parents).copy()
            youngs = self.mutation(youngs).copy()
            self.population = self.survival_selection(youngs).copy()
            for i in self.population:
                print(i.net, i.fitness)
            self.calculate_fitness_avg()
            self.current_iter += 1
            best_current = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[0]
            print(f"current iteration: {self.current_iter} / {self.n_iter}",
                  f", best fitness: {best_current.fitness}, average fitness: {self.fitness_avg}")
            print(f'Network: {best_current.net}')
            print("-------------------------------------------------------------------------------------------------")
            self.fitness_history.append(self.fitness_avg)

        ans = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[0]

        return ans, self.fitness_history