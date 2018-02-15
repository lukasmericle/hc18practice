import numpy as np
from manip import to_rectangles

def get_fitness(chrom, pizza, L, H, alpha, beta, gamma, mu):
    R = to_rectangles(chrom)
    overlap_matrix = np.zeros_like(pizza)
    
    fitness = 0
    for r_i,r_e,c_i,c_e in R:
        slice_matrix = pizza[r_i:r_e+1,c_i:c_e+1];
    
        n_tomatoes = np.sum(slice_matrix)
        n_mushrooms = slice_matrix.size - n_tomatoes
        
        fitness = fitness + alpha*(max(0,L-n_tomatoes))**2 
        fitness = fitness + beta*(max(0,L-n_mushrooms))**2
        fitness = fitness + gamma*(min(0,H-n_mushrooms-n_tomatoes))**2
        
        
        overlap_matrix[r_i:r_e+1,c_i:c_e+1] += 1
        
    n_covered = len(np.where(overlap_matrix>0)[0])
    fitness = fitness - n_covered
    overlap_penalty = mu*np.sum((overlap_matrix-1)[np.where(overlap_matrix>1)])
    fitness = fitness + overlap_penalty
    
    return fitness
