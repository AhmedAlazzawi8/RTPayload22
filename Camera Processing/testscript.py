import numpy as np
import cv2
import copy
import sys
import time
from undistort import undistort
from square_decision import gen_aggregate_matrix, gen_fixed_coords, gen_scale_factor, gen_theta, gen_transform_matrix
from matplotlib import pyplot as plt
from PIL import Image
import datetime
import math

CONSTX = 500
CONSTY = 500

SMALLEST = "book3.jpg"

MIDDLEST = "book2.jpg"

BIGGEST = "book1.jpg"

def sanityCheck():
    sift = cv2.SIFT_create()
    print(cv2.__version__)
        # find the keypoints and descriptors with SIFT
        #kp1, des1 = sift.detectAndCompute(img1,None)
        #kp2, des2 = sift.detectAndCompute(img2,None) 
    pointsAndDescriptors = []
    for i in range(1, 3):
        img_name = "{}.jpg".format(i)
        img = cv2.imread(img_name,0)
        kp, des = sift.detectAndCompute(img,None)
        pointsAndDescriptors.append((kp, des))
        #print(len(kp))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    print(len(pointsAndDescriptors))
    matches = flann.knnMatch(pointsAndDescriptors[0][1], pointsAndDescriptors[1][1], k=2)
    #print(matches)
    
    good_matches = []
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            print(m)
            good_matches.append(m)
     

    from_to = {}

    for m in good_matches: #For loop which puts our matches as coordinate pairs into respective arrays
        from_to[pointsAndDescriptors[0][0][m.queryIdx].pt] = pointsAndDescriptors[1][0][m.trainIdx].pt
        #smaller.append(pointsAndDescriptors[0][0][m.queryIdx].pt)
        #larger.append(pointsAndDescriptors[1][0][m.trainIdx].pt)
        # if (pointsAndDescriptors[0][0][m.queryIdx].pt == None) ^ (pointsAndDescriptors[1][0][m.trainIdx].pt == None):
        #     print(":( points don't all have corresponding point")
    
    #TODO Sort from_to based on distance so best match is first key pair, second best is second etc

    ###
    #Larger image is always first, coordinates on larger image are key to coords on smaller image

    (translate_x, translate_y) = list(from_to.keys())[0] 

    (translate_x_small, translate_y_small) = from_to[list(from_to.keys())[0]]
    
    calculationCoords = gen_fixed_coords(from_to, translate_x, translate_y, translate_x_small, translate_y_small)
    #print(calculationCoords)
    scaleFactor = gen_scale_factor(calculationCoords)
    theta = gen_theta(calculationCoords)
    #print(smaller)
    #print(larger)

    smallToLarge = gen_transform_matrix(-translate_x_small, -translate_y_small, theta, scaleFactor, translate_x, translate_y)

    transformed = []
    avgDistance = 0
    print(smallToLarge)
    print(list(from_to.keys())[0], from_to[list(from_to.keys())[0]])
    print(list(from_to.keys())[0], np.matmul(smallToLarge, np.array([[from_to[list(from_to.keys())[0]][0]], [from_to[list(from_to.keys())[0]][1]], [1]])   ))
    for key in from_to.keys():
        #print(np.matmul( smallToLarge, np.array([[from_to[key][0]],[ from_to[key][1]], [1]])))
        transformed.append(np.matmul( smallToLarge, np.array([[from_to[key][0]],[ from_to[key][1]], [1]])))

    for i in range(len(transformed)):
        #transformed[i]
        print("\n\n")
        print(list(from_to.keys())[i], transformed[i])
        print("\n")
        print()
        #print(np.sqrt((list(from_to.keys())[i][0] -transformed[i][0][0] )** 2 + (list(from_to.keys())[i][1] - transformed[i][1][0]) ** 2))
        #avgDistance = avgDistance + np.sqrt((list(from_to.keys())[i][0] -transformed[i][0][0] )** 2 + (list(from_to.keys())[i][1] - transformed[i][1][0]) ** 2)
    
    #print(avgDistance/ range(len(list(from_to.keys()))))
    (coordx1, coordy1) = list(from_to.keys())[1]
    (coordx2, coordy2) = from_to[list(from_to.keys())[1]]

    #testCoords = np.array([[CONSTX], [CONSTY], [1]])
    testCoordsOG = np.array([[CONSTX], [CONSTY], [1]])
    (coordxTest, coordyTest) = from_to[list(from_to.keys())[0]]

    print("test gen transform no trans", gen_transform_matrix())
    testCoords = [CONSTX - coordxTest, CONSTY - coordyTest, 1] #Small image coords
    
    testMat1 = gen_transform_matrix(translate_x=-coordxTest, translate_y=-coordyTest)
    print("First transform testCoords:", testCoords, "\nGenerated Matrix: ", testMat1, "\nMatrix applied: ", np.matmul(testMat1, testCoordsOG))
    seqMatCoords = np.matmul(testMat1, testCoordsOG)
    #testMat1 = gen_aggregate_matrix
    testCoords[0] = testCoords[0] * scaleFactor
    testCoords[1] = testCoords[1] * scaleFactor
    
    testMat2 = gen_transform_matrix(translate_x = -coordxTest, translate_y = -coordyTest, scale_factor=scaleFactor)
    print("\nChecking for scale")
    print("\nManual Test Coords: ", testCoords)
    print("\nGenerated testMatrix: ", testMat2)
    print("\nApplied test matrix: ", np.matmul(testMat2, testCoordsOG))


    testCoords[0] = testCoords[0]*np.cos(theta) - testCoords[1]*np.sin(theta)
    testCoords[1] = testCoords[0] * np.sin(theta) + testCoords[1] * np.cos(theta)


    print("\nChecking Rotation, theta: ", theta)
    print("\nManual test Coords", testCoords)
    testRot = gen_transform_matrix(translate_x= -coordxTest, translate_y= - coordyTest, scale_factor = scaleFactor, theta=theta)
    print("\nGenerated Matrix: ", testRot)
    print("\nApplied Test Matrix: ", np.matmul(testRot, testCoordsOG))
    
    testCoords[0] = testCoords[0] + list(from_to.keys())[0][0]
    testCoords[1] = testCoords[1] + list(from_to.keys())[0][1]

    print("Manual coords final translation: ", testCoords)
    testFinal = gen_transform_matrix(translate_x= -coordxTest, translate_y= - coordyTest, scale_factor = scaleFactor, theta=theta,
     translate_x_large=list(from_to.keys())[0][0], translate_y_large=list(from_to.keys())[0][1])

    print("\nGenerated matrix: ", testFinal)
    print("\nApplied matrix: ", np.matmul(testFinal, testCoordsOG))

def image_registration_matrix(img_name1 : str, img_name2 : str):
    #Returns transform matrix between the two images
    sift = cv2.SIFT_create()

    img1 = cv2.imread(img_name1, 0)
    img2 = cv2.imread(img_name2, 0)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            good_matches.append(m)

    from_to = {}

    for m in good_matches: #For loop which puts our matches as coordinate pairs into respective arrays
        from_to[kp1[m.queryIdx].pt] = kp2[m.trainIdx].pt

    #TODO Sort matches based on quality or metric like distance

    (translate_x, translate_y) = list(from_to.keys())[0] 
    #(coordxTest, coordyTest) = from_to[list(from_to.keys())[0]]
    (translate_x_small, translate_y_small) = from_to[list(from_to.keys())[0]]
    
    calculationCoords = gen_fixed_coords(from_to, translate_x, translate_y, translate_x_small, translate_y_small)
    
    scaleFactor = gen_scale_factor(calculationCoords)
    theta = gen_theta(calculationCoords)

    return gen_transform_matrix(translate_x= -translate_x_small, translate_y= -translate_y_small, scale_factor = scaleFactor, theta=theta,
     translate_x_large=list(from_to.keys())[0][0], translate_y_large=list(from_to.keys())[0][1])


def camera():
    cap = cv2.VideoCapture(0)
    #print("What the fuck")
    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cap.read()
        #if not ret:
         #   print("failed to grab frame")
          #  continue
        frame = undistort(frame)
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cap.release()
    




if __name__ == "__main__":
    
    #TODO Test surf by installing other opencv
    #surf = cv2.xfeatures2d.SURF_create(400)
   
    testCoordsOG = np.array([[CONSTX], [CONSTY], [1]])
    twoToOne = image_registration_matrix(BIGGEST, MIDDLEST)
    threeToTwo = image_registration_matrix(MIDDLEST, SMALLEST)
    threeToOne = gen_aggregate_matrix(twoToOne, threeToTwo)

    #print("\n\nGenerated matrix registration: ", np.matmul(twoToOne, testCoordsOG))
    

    img3 = cv2.imread(SMALLEST)
    img3 = cv2.circle(img3, (math.floor(CONSTX), math.floor(CONSTY)), 7, (255, 0, 0), 5)
    cv2.imshow("thing3", img3)



    fromThree = np.matmul(threeToTwo, testCoordsOG)
    print(fromThree)
    #print(fromThree[0, 0])
    #print(fromThree[1, 0])


    img2 = cv2.imread(MIDDLEST)
    img2 = cv2.circle(img2, (math.floor(fromThree[0, 0]), math.floor(fromThree[1, 0])), 7, (255, 0, 0), 5)
    cv2.imshow("thing2", img2)

    fromOG = np.matmul(threeToOne, testCoordsOG)
    fromTwo = np.matmul(twoToOne, fromThree)
    
    print(fromOG)
    
    img1 = cv2.imread(BIGGEST)
    img1 = cv2.circle(img1, (math.floor(fromOG[0, 0]), math.floor(fromOG[1, 0])), 7, (0, 0, 255), 5)
    img1 = cv2.circle(img1, (math.floor(fromTwo[0, 0]), math.floor(fromTwo[1, 0])), 30, (255, 0, 0), 5)
    cv2.imshow("thing", img1)

    
    




    cv2.waitKey(0)
    cv2.destroyAllWindows()