# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:30:15 2018

@author: andre
"""

# [ri,re,ci,ce]
import numpy as np

def get_fitness(R, pizza, L, H, alpha, beta, gamma, mu):

    overlap_matrix = np.zeros_like(pizza)
    
    fitness = 0
    for r_i,r_e,c_i,c_e in R:
        slice_matrix = pizza[r_i:r_e+1,c_i:c_e+1];
    
        n_tomatoes = np.sum(slice_matrix)
        n_mushrooms = slice_matrix.size - n_tomatoes
        
        fitness = fitness + alpha*(max(0,L-n_tomatoes))**2 
        fitness = fitness + beta*(max(0,L-n_mushrooms))**2
        fitness = fitness + gamma*(min(0,H-n_mushrooms-n_tomatoes))**2
        fitness = fitness - (r_e-r_i+1)*(c_e-c_i+1)
        
        slice_big_matrix = np.zeros(pizza.shape)
        slice_big_matrix[r_i:r_e+1,c_i:c_e+1] = 1
        overlap_matrix = overlap_matrix + slice_big_matrix
        

    fitness = fitness + mu*np.sum((overlap_matrix-1)[np.where(overlap_matrix>1)])
    
    return fitness
