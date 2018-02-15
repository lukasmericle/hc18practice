import numpy as np
from random import random

def create_rectangles(n, rows, cols):
    rs = np.random.random((n,4))
    rs[:,[0,2]] *= rows
    rs[:,[1,3]] *= cols
    return np.int16(rs)

def to_chromosome(rs):
    return rs.flatten()

def rshp(chrom):
    l = len(chrom)
    n = l - l%4
    return n, chrom[:n].reshape((-1,4))

def to_rectangles(chrom):
    n, rs = rshp(chrom)
    r1 = np.minimum(rs[:,0], rs[:,2]).reshape((-1,int(n/4)))
    r2 = np.maximum(rs[:,0], rs[:,2]).reshape((-1,int(n/4)))
    c1 = np.minimum(rs[:,1], rs[:,3]).reshape((-1,int(n/4)))
    c2 = np.maximum(rs[:,1], rs[:,3]).reshape((-1,int(n/4)))
    rs = np.hstack((r1.T, c1.T, r2.T, c2.T))
    return rs

def to_submission(chrom, title):
    rs = to_rectangles(chrom)
    n = rs.shape[0]
    f = open("submission_"+title+".txt", "w")
    f.write(str(n)+'\n')
    for r in rs:
        r1 = r[0]; c1 = r[1]; r2 = r[2]; c2 = r[3]
        f.write(str(r1) + ' ' + str(c1) + ' ' + str(r2) + ' ' + str(c2) + '\n')
    f.close()

def get_overlap(chrom, pizza):
    R = to_rectangles(chrom)
    overlap_matrix = np.zeros_like(pizza)
    for r_i,c_i,r_e,c_e in R:
        overlap_matrix[r_i:r_e+1,c_i:c_e+1] += 1
    print()
    print(overlap_matrix, R.shape[0])