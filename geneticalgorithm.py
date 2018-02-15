import numpy as np
from numpy.random import randint
from random import random
from ComputeSingleFitness import get_fitness

def individual_cross_parts(input_string, cross_point):
    first_part  = input_string[:cross_point]
    second_part = input_string[cross_point:]
    return first_part, second_part

def crossover(input_string1, input_string2):
    n_instructions1 = len(input_string1)
    n_instructions2 = len(input_string2)
    
    cross_points = np.zeros(2)
        
    cross_points[0] = randint(3, n_instructions1)
    cross_points[1] = randint(3, n_instructions2)
    cross_points = cross_points.astype(int)    
    
    chromosome1_front, chromosome1_end = individual_cross_parts(input_string1, cross_points[0])
    chromosome2_front, chromosome2_end = individual_cross_parts(input_string2, cross_points[1])  
    
    new_individual1 = np.append(chromosome1_front, chromosome2_end)
    new_individual2 = np.append(chromosome2_front, chromosome1_end)
    
    return new_individual1, new_individual2
    
def select(population, fitnesses, n_indiviudals):
    parents = []
    probs = np.cumsum(fitnesses)/np.sum(fitnesses)
    for i in range(n_indiviudals):
        r = random()
        k = 0
        while r > probs[k]:
            k += 1
        parents.append(population[k])
    return parents

def mutate(chrom, rows, cols, sigma):
    mu = 0
    randos = np.random.normal(mu, sigma, len(chrom))
    chrom += np.around(randos)
    chrom[:,[0,2]] = np.max(0, np.min(rows, chrom[:,[0,2]]))
    chrom[:,[1,3]] = np.max(0, np.min(cols, chrom[:,[1,3]]))
    return chrom

def find_elite(chroms, fitnesses, k):
    elites = fitnesses.argsort()[:k]
    return chroms[elites]

def gen_population(rows, cols, min_rectangles, max_rectangles):
    population = []
    for i in population_size:
        n = int(random() * (max_rectangles - min_rectangles) + min_rectangles)
        chrom = to_chromosome(create_rectangles(n, rows, cols))
        population.append(chrom)
    return population

def get_fitnesses(chroms, pizza, L, H, alpha, beta, gamma, mu):
    fits = np.zeros(len(chroms))
    for i,chrom in enumerate(chroms):
        fits[i] = get_fitness(chrom, pizza, L, H, alpha, beta, gamma, mu)
    return fits


def write_best_to_file(population, fits, title):
    bestfit = fits.argsort()[0]
    bestchrom = population[bestfit]
    to_submission(bestchrom, title)


def ga_loop(pizza, title, constraints, n_generations=100, population_size=10, n_elite=1,
            crossover_prob=0.1, mutation_prob=0.5):
    
    rows, cols = pizza.shape
    L, H = constraints
    
    alpha = 1
    beta = 1
    gamma = 1
    mu = 1
    sigma = 0.1*(rows+cols)/2

    min_rectangles, max_rectangles = get_opt_number_rectangles()
    population = gen_population(rows, cols, min_rectangles, max_rectangles)
    fitnesses = np.zeros(n_generations, population_size)

    for gen in range(n_generations):
        
        fitnesses[gen,:] = get_fitnesses(population, pizza, L, H, alpha, beta, gamma, mu)
        if gen%10==0:
            write_best_to_file(population, fitnesses[gen,:], title)

        new_population = find_elite(population, fitnesses[gen,:], n_elite)

        for i in range(population_size - n_elite):
            if random() < crossover_prob:
                parent1, parent2 = select(population, fitnesses[gen,:], 2)
                child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        for i in range(n_elite, population_size):
            if random() < mutation_prob:
                new_population[i] = mutate(new_population[i], rows, cols, sigma)
        
        population = new_population
        sigma *= 0.999
        alpha *= 1.001
        beta  *= 1.001
        gamma *= 1.001
        mu    *= 1.01

    final_fitnesses = get_fitnesses(population, pizza, L, H, alpha, beta, gamma, mu)
    write_best_to_file(population, fitnesses[gen,:], title)
