import numpy as np
from filehandler import get_problem

problem = "example.in"
pizza, minIngredients, maxSize = get_problem(problem)
print(pizza)

def create_rectangles(n, rows, cols):
    rs = np.random.random((n,4))
    rs[:,[0,1]] *= rows
    rs[:,[2,3]] *= cols
    return np.int8(rs)

def to_chromosome(rs):
    return rs.flatten()

def to_rectangles(rs):
    return rs.reshape((-1,4))