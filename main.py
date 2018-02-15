import numpy as np
from random import random
from filehandler import get_problem

problem = "example.in"
pizza, minIngredients, maxSize = get_problem(problem)
print(pizza)

def create_rectangles(n, rows, cols):
    rs = np.random.random((n,4))
    rs[:,[0,2]] *= rows
    rs[:,[1,3]] *= cols
    return np.int(rs)

def to_chromosome(rs):
    return rs.flatten()

def to_rectangles(chrom):
    l = len(chrom)
    n = l - l%4
    rs = chrom[:n].reshape((-1,4))
    r1 = np.minimum(rs[:,0], rs[:,2])
    r2 = np.maximum(rs[:,0], rs[:,2])
    c1 = np.minimum(rs[:,1], rs[:,3])
    c2 = np.maximum(rs[:,1], rs[:,3])
    rs = np.hstack((r1.T, c1.T, r2.T, c2.T)) # not sure if transpose works properly--may need to make each one a 2D array first
    return rs
        
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

def ga_loop(pizza, n_generations=100, population_size=10, n_elite=1,
            crossover_prob=0.1, mutation_prob=0.5):
    
    rows, cols = pizza.shape
    
    min_rectangles, max_rectangles = get_opt_number_rectangles()
    population = gen_population(rows, cols, min_rectangles, max_rectangles)
    fitnesses = np.zeros(n_generations, population_size)

    for gen in n_generations:
        for i in range(population_size):
            fitnesses[gen,i] = get_fitness(chrom, pizza)
        
        new_population = find_elite(population, fitnesses[gen,:], n_elite)
        
        for i in range(population_size - n_elite):
            if random() < crossover_prob:
                parent1, parent2 = select(population,fitnesses[gen,:],2)
                child = crossover(parent1, parent2)
            else:
                child = select(population,fitnesses[gen,:],1)
            if random() < mutation_prob:
                child = mutate(child, rows, cols, sigma)
            new_population.append(child)
            
        population = new_population
    
            
