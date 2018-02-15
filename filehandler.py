import numpy as np

TOMATO_CHAR = 'T'

def get_problem(filename):
    f = open(filename)
    header = f.readline()
    nums = header.split()
    
    rows = int(nums[0])
    cols = int(nums[1])
    minIngredients = int(nums[2])
    maxSize = int(nums[3])
    
    pizza = np.zeros((rows,cols))
    
    for row in range(rows):
        line = f.readline()
        tomatoes = [pos for pos, char in enumerate(line) if char == TOMATO_CHAR]
        pizza[row,tomatoes] = 1
    
    return pizza, minIngredients, maxSize