import numpy as np
import cv2
import copy
import sys
from matplotlib import pyplot as plt
from PIL import Image


# Alternative to drawMatches. We can probably use this to find out how to interpret the matches
# Source: https://gist.github.com/isker/11be0c50c4f78cad9549
def draw_matches_on_img2(img1, kp1, img2, kp2, matches, color=None):     
    new_img = np.zeros(img2.shape, type(img2.flat[0]))  
    new_img[0:img2.shape[0],0:img2.shape[1]] = img2

    r = 10
    thickness = 2
    if color:
        c = color
    else: 
        c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)

    for m in matches:
        cv2.circle(new_img, tuple(np.round(kp2[m.trainIdx].pt).astype(int)), r, c, thickness)

    return new_img



def calculateError(smaller, larger):
#    distance1 = np.sqrt(((smaller[a][0] - smaller[b][0]) ** 2 + (smaller[a][1] - smaller[b][1]) ** 2))

    #square each matrix subtract one from other, then square root (distances summed)
    difference = np.add.reduce(np.sqrt(np.add.reduce(np.square(np.subtract(larger,smaller)), 1)), 0)
    return difference

    

def scalePoints(smaller, larger, a, b):

    #Distance calculations for small image vector and big image vector
    distance1 = np.sqrt(((smaller[a][0] - smaller[b][0]) ** 2 + (smaller[a][1] - smaller[b][1]) ** 2))
    distance2 = np.sqrt(((larger[a][0] - larger[b][0]) ** 2 + (larger[a][1] - larger[b][1]) ** 2))
    
    
    scale_factor = distance2/distance1
        
    
    smaller = np.multiply(smaller, scale_factor) #Scale points
    
    return smaller, scale_factor
    
    
def rotatePoints(smaller, larger, a, b):    
    
    smaller = np.subtract(smaller, smaller[a]) #Translate smaller image points so best match is origin
    larger = np.subtract(larger, larger[a]) #Translate larger image points so best match is origin

    theta = np.arctan2(smaller[b][0], smaller[b][1]) - np.arctan2(larger[b][0], larger[b][1])
    smaller = [[war[0]*np.cos(theta) - war[1]*np.sin(theta), war[0] * np.sin(theta) + war[1] * np.cos(theta)] for war in smaller]    
    
    return smaller, theta



def process_matches(smaller, larger, a, b):     
    smaller_copy = copy.deepcopy(smaller)
    
    #Process the points
    #See this!!!: https://math.stackexchange.com/questions/1544147/find-transform-matrix-that-transforms-one-line-segment-to-another
    smaller_copy, scale_factor  = scalePoints(smaller_copy, larger, a, b)
    smaller_copy, rotate_factor = rotatePoints(smaller_copy, larger, a, b)
    smaller_copy, translation_factor = np.add(smaller_copy, larger[a]), larger[a] #translate back to original points
    
    return smaller_copy, scale_factor, rotate_factor, translation_factor



def optimizePointSelection(img1, kp1, img2, kp2, matches):
    #Check for enough matches
    if len(matches) < 2:
        print("Error, not enough matches. Two Required, %d given\n" % len(matches))
        raise "ERROR"
        return None    
    
    
    #Split matches into points from each image
    smaller = [] 
    larger = []
   
    for m in matches: #For loop which puts our matches as coordinate pairs into respective arrays
        smaller.append(kp1[m.queryIdx].pt)
        larger.append(kp2[m.trainIdx].pt)
        if (kp1[m.queryIdx].pt == None) ^ (kp2[m.trainIdx].pt == None):
            print(":( points don't all have corresponding point")
    
    
    errorList = [] #contains a list of tuples that contain the error and a tuple containing the endpoint indices for the line (ex: [(#, (a,b))])
    
    
    #Choose reference points
    #These are the two indexes of points that form a line from each plot that are used for calculation
    
    for a in range(len(matches)):
        for b in range(a, len(matches)): #double check the "a," in the range later
            if a == b:
                continue
            temp_smaller, _, _, _ = process_matches(smaller, larger, a, b)
            errorList.append((calculateError(temp_smaller, larger), (a,b)))


    
    a_optimal, b_optimal = min(errorList)[1]
    
    
    
    smaller, scale_factor, rotate_factor, translation_factor = process_matches(smaller, larger, a_optimal, b_optimal)
    
    print("\nOptimal Error: ", calculateError(smaller, larger))
    
    #show results
       
    
    #draw image
    #plt.imshow(img2)
    
    #draw other image with the translation, rotation, scaling, etc
    

    # dividing height and width by 2 to get the center of the image
    height, width = img1.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-rotate_factor*180/3.1415926, scale=scale_factor)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=img1, M=rotate_matrix, dsize=(width, height))
    
    

    image = Image.open(sys.argv[1])
    # logo = Image.open('11.png')
    # logo = logo.convert("RGBA")
    image_copy = image.copy()

    #image_copy.paste(logo, position)
    image_copy.paste(img1, translation_factor, img1)
    
    plt.imshow(image_copy)

    #draw points
    
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')

    plt.plot(smaller[a_optimal][0], smaller[a_optimal][1], 'og')
    plt.plot(smaller[b_optimal][0], smaller[b_optimal][1], 'og')
    plt.plot(larger[a_optimal][0], larger[a_optimal][1], '*y')
    plt.plot(larger[b_optimal][0], larger[b_optimal][1], '*y')

    plt.show()
    plt.clf()
    
    
    
    
    #Show worst possible 
    a_optimal, b_optimal = max(errorList)[1]
        
    smaller, scale_factor, rotate_factor, translation_factor = process_matches(smaller, larger, a_optimal, b_optimal)
    
    print("\nOptimal Error: ", calculateError(smaller, larger))
    
    #show results
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')

    plt.plot(smaller[a_optimal][0], smaller[a_optimal][1], 'og')
    plt.plot(smaller[b_optimal][0], smaller[b_optimal][1], 'og')
    plt.plot(larger[a_optimal][0], larger[a_optimal][1], '*y')
    plt.plot(larger[b_optimal][0], larger[b_optimal][1], '*y')

    plt.show()
    plt.clf()
    
    
    return 

MIN_MATCH_COUNT = 1

print("Did you enter the image paths as args? do python keypointtest.py <larger> <smaller>")

print(sys.argv[1], sys.argv[2])

img2 = cv2.imread(sys.argv[1])      # queryImage
img1 = cv2.imread(sys.argv[2])             # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
#The lower the value, the less points... I think
# CONSTANT_TEST Determines sensitivity of whats a good match
# CONSTANT_TEST = 0.7 # <-- Default
CONSTANT_TEST = 0.6 #Testing, .8 seems good. 

good = []
for m,n in matches:
    if m.distance < CONSTANT_TEST*n.distance:
        good.append(m)

print("There are %d good matches" % len(good))

    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                #    matchesMask = matchesMask, # draw only inliers
                   matchesMask= None,
                   flags = 2)

img3 = cv2.drawMatches(img1, kp1, img2,kp2,good[-3:-1],None,**draw_params)

img4 = draw_matches_on_img2(img1, kp1, img2, kp2, good, color=(255,0,0))


plt.imshow(img4)
print("Press Q to see other/quit")
plt.show()


plt.imshow(img3)
plt.show()
optimizePointSelection(img1, kp1, img2, kp2, good)

#Testing error function
error_test_smaller = [(0,0), (1,1), (3,4)]
error_test_larger = [(0,0), (1,0), (0,0)]

print("Smaller: ", error_test_smaller, "\nLarger: ", error_test_larger, "\nError: ", calculateError(error_test_smaller, error_test_larger))
print("It works!!!")

