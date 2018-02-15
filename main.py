import numpy as np
from filehandler import get_problem

problem = "example.in"
pizza, minIngredients, maxSize = get_problem(problem)
print(pizza)

def create_rectangles(n, rows, cols):
    rs = np.random.random((n,4))
    rs[:,[0,1]] *= rows
    rs[:,[2,3]] *= cols
    return np.int16(rs)

def to_chromosome(rs):
    return rs.flatten()

def to_rectangles(chrom):
    l = length(chrom)
    n = int(l/4) - l%4
    rs = chrom[:n].reshape((-1,4))
    rs = chrom.reshape((-1,4))
    for i in range(rs.shape[0]):
        #r1 = rs[i,0]
        #r2 = rs[i,1]
        #c1 = rs[i,2]
        #c2 = rs[i,3]
        if rs[i,0] > rs[i,1]:
            t = rs[i,1]
            rs[i,1] = rs[i,0]
            rs[i,0] = t
        if rs[i,2] > rs[i,3]:
            t = rs[i,3]
            rs[i,3] = rs[i,2]
            rs[i,2] = t
    return rs
        