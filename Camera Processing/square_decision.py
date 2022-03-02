import sys

import numpy as np

#20x20 grid
"""
1,1  | 2,1  | 3,1  | ... | 1, 1

...

...

1,20 | 2,20 | 3,20 | ... | 20, 20

"""
def guess(x_coord: int, y_coord:int) -> int:
    
    foundX = false
    foundY = false
    
    xGuess = 13
    yGuess = 13
    for i in range(1, 20):
        if( 250 * i > x_coord && !foundX):
            xGuess = i

        if(250 * i > y_coord && !foundY):
            yGuess = i

    return(xGuess, yGuess)

def gen_coords(x_coord: float, y_coord: float, transformation_matrices):
    """
    Takes coordinates as input, outputs location relative to the original image

    param x_coord: x coordinate from image we want to convert into original coordinate system
    param y_coord: y coordinate from image we want to convert into original coordinate system

    param tramsformation_matrices: array transformation matrices in order of application (smallest to second smallest, etc) until we are back on image with orignal grid.
    """

    a = np.matrix([x_coord, y_coord, 0.0])

    for matrix in transformation_matrices:
        a = np.matmul(a, matrix)    
    
    return(a.item(0,0), a.item(0,1))

#simple test method
if __name__ == "__main__":
    
    for arg in sys.argv:
        print(arg)

    print(guess(sys.argv[0], sys.argv[1]))


