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
    for i in range(rs.shape[0]):
        #r1 = rs[i,0]
        #c1 = rs[i,1]
        #r2 = rs[i,2]
        #c2 = rs[i,3]
        if rs[i,0] > rs[i,2]:
            t = rs[i,2]
            rs[i,2] = rs[i,0]
            rs[i,0] = t
        if rs[i,1] > rs[i,3]:
            t = rs[i,3]
            rs[i,3] = rs[i,1]
            rs[i,1] = t
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