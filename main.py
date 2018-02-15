import numpy as np
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
    chrom += randos
    chrom[:,[0,2]] = np.max(0, np.min(rows, chrom[:,[0,2]]))
    chrom[:,[1,3]] = np.max(0, np.min(cols, chrom[:,[1,3]]))
    return chrom

def find_elite(chroms, fitnesses, k):
    elites = fitnesses.argsort()[:k]
    return chroms[elites]