import numpy as np

MUSHROOM_CHAR = 'M'

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
        mushrooms = [pos for pos, char in enumerate(line) if char == MUSHROOM_CHAR]
        pizza[row,mushrooms] = 1
    
    return pizza, minIngredients, maxSize