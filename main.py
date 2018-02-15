import numpy as np
from random import random
from filehandler import get_problem
from geneticalgorithm import *

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

def to_submission(chrom, title):
    rs = to_rectangles(chrom)
    n = rs.shape[0]
    f = open("submission_"+title+".txt", "w")
    f.write(str(n)+'\n')
    for r in rs:
        r1 = r[0]; c1 = r[1]; r2 = r[2]; c2 = r[3]
        f.write(str(r[0]) + ' ' + str(r[1]) + ' ' + str(r2) + ' ' + str(c2) + '\n')
    f.close()
