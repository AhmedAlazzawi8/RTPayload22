import sys

import numpy as np

def gen_transform_matrix(translate_x, translate_y, theta, scale_factor):

    translate_matrix = np.matrix([[1, 0, translate_x], [0, 1, translate_y], [0, 0, 1]])
    rotation_matrix = np.matrix([[np.cos(theta), -1*np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    scale_matrix = np.matrix([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])

    #scale, rotate, translate but in reverse so translate, rotate, scale is the order to multiply

    transform_matrix = np.matmul(translate_matrix, rotation_matrix)
    
    print("\nTransform matrix before multiplying by scale: ", transform_matrix, "\n")
    return np.matmul(transform_matrix, scale_matrix)

def gen_aggregate_matrix(larger_to_map, smaller_to_larger):

    return np.matmul(smaller_to_larger, larger_to_map)

#20x20 grid
"""
1,1  | 2,1  | 3,1  | ... | 1, 1

...

...

1,20 | 2,20 | 3,20 | ... | 20, 20

"""
def guess(x_coord: float, y_coord: float) -> int:
    ###

    #TODO: Make guess correspond to the actual image, eg define the drid in terms of coordinate system, in fact, define entire image in terms of coordinate system
    
    ###
    foundX = False
    foundY = False
    
    xGuess = 13
    yGuess = 13
    for i in range(1, 20):
        if( 250 * i > float(x_coord) and not foundX):
            xGuess = i
            foundX = True

        if(250 * i > float(y_coord) and not foundY):
            yGuess = i
            foundY = True

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
    
    for index in range(len(sys.argv)):
        print(sys.argv[index])

    
    print(guess(float(sys.argv[1]), float(sys.argv[2])))

    print("\nSanity check for some matrix math\n")
    
    testCoords = np.matrix([[1],[ 1], [1]])
    #scale 
    scaleTest = np.matrix([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    
    testCoords = np.matmul(scaleTest, testCoords);
    #rotate
    rotateTest = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    testCoords = np.matmul(rotateTest, testCoords);
    print(testCoords)
    #translate
    translateTest = np.matrix([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
    
    testCoords = np.matmul(translateTest, testCoords)

    print("Test coords: ", testCoords)


    tester = gen_transform_matrix(3, 4, np.pi*3/2, 2)
    print(tester)
    print("Test transform: ", np.matmul(tester, [[1],[ 1], [1]]))
    #We don't have to worry about the preceding translation to the origin because it's applied to every point within the image 
