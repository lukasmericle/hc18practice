import numpy as np
from numpy.random import randint, binomial
from random import random
from ComputeSingleFitness import get_fitness
from manip import *

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

def tournselect(population, fitnesses, n_individuals, ptournament=0.75, tournament_size=4):
    parents = []
    for i in range(n_individuals):
        tourn_pop = []
        fitness_vals = []
        for i in range(tournament_size):
            temp_individual_index = randint(0,len(population))
            tourn_pop.append(temp_individual_index)
            fitness_vals.append(fitnesses[temp_individual_index])
        individual_was_selected = False
        while(individual_was_selected == False):
            r = random()
            min_fitness_index = np.argmin(fitness_vals)
            if r < ptournament:
                parents.append(population[tourn_pop[min_fitness_index]])
                individual_was_selected = True
            else:
                fitness_vals[min_fitness_index] = 1e6
    return parents

def mutate(chrom, rows, cols, sigma, mut_per_gene):
    mu = 0
    randos = np.random.normal(mu, sigma, len(chrom))
    chrom += np.around(randos * binomial(1, mut_per_gene, len(chrom))).astype(np.int16)
    n, rs = rshp(chrom)
    zeros = np.zeros((rs.shape[0],2))
    rs[:,[0,2]] = np.maximum(zeros, np.minimum(zeros+rows, rs[:,[0,2]]))
    rs[:,[1,3]] = np.maximum(zeros, np.minimum(zeros+cols, rs[:,[1,3]]))
    # latent bug: does not rescale last 1,2,3 genes if they are present
    rechrom = to_chromosome(rs)
    return np.append(rechrom, chrom[len(chrom)%4:])

def find_elite(chroms, fitnesses, k):
    elites = fitnesses.argsort()[:k]
    return [el for i,el in enumerate(chroms) if i in elites]

def gen_population(rows, cols, population_size, min_rectangles, max_rectangles):
    population = []
    for i in range(population_size):
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

def get_opt_number_rectangles(L, H, n_mushrooms, n_tomatoes):
    nr1 = np.floor(n_mushrooms/L)
    nr2 = np.floor(n_tomatoes/L)
    nr3 = np.ceil((n_mushrooms + n_tomatoes)/H)
    nmin = min([nr1, nr2])
    nmax = max([nmin, nr3])
    return nmin, nmax

def ga_loop(pizza, title, constraints, n_generations=100, population_size=10, n_elite=1,
            crossover_prob=0.1, mutation_prob=1):
    
    rows, cols = pizza.shape
    L, H = constraints

    n_mushrooms = np.sum(pizza)
    n_tomatoes = pizza.size - n_mushrooms
    
    alpha = 10
    beta = 10
    gamma = 10
    mu = 100
    sigma = 0.1*(rows+cols)/2

    min_rectangles, max_rectangles = get_opt_number_rectangles(L, H, n_mushrooms, n_tomatoes)
    avg_rectangles = (min_rectangles+max_rectangles)/2
    mut_per_gene = mutation_prob/avg_rectangles
    population = gen_population(rows, cols, population_size, min_rectangles, max_rectangles)
    fitnesses = np.zeros((n_generations, population_size))

    for gen in range(n_generations):
        
        fitnesses[gen,:] = get_fitnesses(population, pizza, L, H, alpha, beta, gamma, mu)
        print(fitnesses[gen,:])
        if gen%10==0:
            write_best_to_file(population, fitnesses[gen,:], title)

        new_population = find_elite(population, fitnesses[gen,:], n_elite)

        for i in range(int((population_size - n_elite)/2)):
            parent1, parent2 = tournselect(population, fitnesses[gen,:], 2)
            if random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)

        for i in range(n_elite, population_size):
            new_population[i] = mutate(new_population[i], rows, cols, sigma, mut_per_gene)
        
        population = new_population
        sigma *= 0.999
        alpha *= 1.001
        beta  *= 1.001
        gamma *= 1.001
        mu    *= 1.01

    final_fitnesses = get_fitnesses(population, pizza, L, H, alpha, beta, gamma, mu)
    write_best_to_file(population, fitnesses[gen,:], title)
