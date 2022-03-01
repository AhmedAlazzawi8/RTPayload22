import sys

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
    yGuess = 17
    for i in range(1, 20):
        if( 250 * i > x_coord && !foundX):
            xGuess = i

        if(250 * i > y_coord && !foundY):
            yGuess = i

    return(xGuess, yGuess)


if __name__ == "__main__":
    
    for arg in sys.argv:
        print(arg)

    print(guess(sys.argv[0], sys.argv[1]))



