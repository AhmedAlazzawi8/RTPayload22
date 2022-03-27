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


if __name__ == "__main__":
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
    
    #surf = cv2.xfeatures2d.SURF_create(400)
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

    CONSTX = 500
    CONSTY = 500
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
    
    testMat2 = gen_transform_matrix(scale_factor=scaleFactor)

    testCoords[0] = testCoords[0]*np.cos(theta) - testCoords[1]*np.sin(theta)
    testCoords[1] = testCoords[0] * np.sin(theta) + testCoords[1] * np.cos(theta)

    testCoords[0] = testCoords[0] + list(from_to.keys())[0][0]
    testCoords[1] = testCoords[1] + list(from_to.keys())[0][1]

    print("Manual test Coords", testCoords)
    
    testCoords = np.matmul(smallToLarge, testCoords)
    print("test coords", testCoords)
    img = cv2.imread("1.jpg")
    #img[coordsx:coordsx+1,coordy:coordy+1] = (0,0,0)
    img = cv2.circle(img, (math.floor(618), math.floor(706)), 7, (0, 0, 255), 5)
    cv2.imshow("thing", img)
    img2 = cv2.imread("2.jpg")
    img2 = cv2.circle(img2, (math.floor(500), math.floor(500)), 7, (0, 0, 255), 5)
    cv2.imshow("thing2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()