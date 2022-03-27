import numpy as np
import cv2
import copy
import sys
import time
from undistort import undistort
from square_decision import gen_fixed_coords, gen_scale_factor, gen_theta, gen_transform_matrix
from matplotlib import pyplot as plt
from PIL import Image
import datetime


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
    for i in range(0, 2):
        img_name = "opencv_frame_{}.png".format(i)
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
    for m,n in matches:
        if m.distance < 0.6*n.distance:
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
    print(calculationCoords)
    scaleFactor = gen_scale_factor(calculationCoords)
    theta = gen_theta(calculationCoords)
    #print(smaller)
    #print(larger)

    smallToLarge = gen_transform_matrix(-translate_x_small, -translate_y_small, theta, scaleFactor, translate_x, translate_y)

    transformed = []
    avgDistance = 0
    print(smallToLarge)
    for key in from_to.keys():
        #print(np.matmul( smallToLarge, np.array([[from_to[key][0]],[ from_to[key][1]], [1]])))
        transformed.append(np.matmul( smallToLarge, np.array([[from_to[key][0]],[ from_to[key][1]], [1]])))

    #for i in range(len(transformed)):
        #transformed[i]
        #print(list(from_to.keys())[i], transformed[i])
        #print(np.sqrt((list(from_to.keys())[i][0] -transformed[i][0][0] )** 2 + (list(from_to.keys())[i][1] - transformed[i][1][0]) ** 2))
        #avgDistance = avgDistance + np.sqrt((list(from_to.keys())[i][0] -transformed[i][0][0] )** 2 + (list(from_to.keys())[i][1] - transformed[i][1][0]) ** 2)
    
    #print(avgDistance/ range(len(list(from_to.keys()))))

    cv2.destroyAllWindows()