from manip import *
from filehandler import get_problem
from geneticalgorithm import *

problem = "example.in"
pizza, L, H = get_problem(problem)
print(pizza)

ga_loop(pizza, problem[:-3], (L,H), n_generations=1000, population_size=25, n_elite=1, crossover_prob=0.4, mutation_prob=1)