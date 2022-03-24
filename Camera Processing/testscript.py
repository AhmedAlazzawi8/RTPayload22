import numpy as np
import cv2
import copy
import sys
import time
from undistort import undistort
#from square_decision as square_decision
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
    
    

    cv2.destroyAllWindows()