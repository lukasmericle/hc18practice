# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:03:16 2018

@author: wdrosko
"""

import numpy as np
from numpy.random import randint

def individual_cross_parts(input_string,cross_point):
    first_part = input_string[0:cross_point]
    second_part = input_string[cross_point:]
    
    return first_part, second_part

def crossover(input_string1, input_string2):
    n_instructions1 = len(input_string1)
    n_instructions2 = len(input_string2)
    
    cross_points = np.zeros(2)
        
    cross_points[0] = randint(3,n_instructions1)
    cross_points[1] = randint(3,n_instructions2)
    cross_points = cross_points.astype(int)    
        
    print(cross_points)
    [chromosome1_front,chromosome1_end] = individual_cross_parts(input_string1,cross_points[0])
    [chromosome2_front,chromosome2_end] = individual_cross_parts(input_string2,cross_points[1])  
    
    new_individual1 = np.append(chromosome1_front,chromosome2_end)
    new_individual2 = np.append(chromosome2_front,chromosome1_end)
    
    return new_individual1, new_individual2
    
#%%