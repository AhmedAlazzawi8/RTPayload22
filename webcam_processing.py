import cv2
import numpy as np
c = cv2.VideoCapture(0)

while(1):
    _,f = c.read()
    
    # processed = cv2.cvtColor(f,  cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(f)
    orb = cv2.ORB_create(200)
    keypoint, des = orb.detectAndCompute(img, None)
    img_final = cv2.drawKeypoints(img, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(img_final)
    # cv2.imshow('e2',processed)

    if cv2.waitKey(5)==27:
        break
cv2.destroyAllWindows()

# Thoughts
# Reverse video, use the last iamge (on the ground) as the one to track,
# Let it run through till apogee,
# compare with grid to find location
# See this? https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# If that does not have enough accuraccy, then for each frame, 
# identify keypoints and compare with the previous frame to find where it is.
# At the end, find the location with the most points, 
# compare with grid to find location

"""
img = cv2.imread('monkey.jpg',0)
orb = cv2.ORB_create(200)
keypoint, des = orb.detectAndCompute(img, None)
img_final = cv2.drawKeypoints(img, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2_imshow(img_final)
"""
