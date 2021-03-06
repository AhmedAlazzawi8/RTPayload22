import numpy as np
import cv2
import copy
import sys
import time
from matplotlib import pyplot as plt
from PIL import Image
import datetime

class Camera:

    def __init__(self):
        self.init()

    # Alternative to drawMatches. We can probably use this to find out how to interpret the matches
    # Source: https://gist.github.com/isker/11be0c50c4f78cad9549
    def draw_matches_on_img2(self, img1, kp1, img2, kp2, matches, color=None):     
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



    def calculateError(self, smaller, larger):
        #square each matrix subtract one from other, then square root (distances summed)
        return np.add.reduce(np.sqrt(np.add.reduce(np.square(np.subtract(larger,smaller)), 1)), 0)




    def scaleX(self, points, scale_factor):      
        points = np.multiply(points, scale_factor) #Scale points
        return points
        
    def rotateX(self, points, theta):    
        points = [[war[0]*np.cos(theta) - war[1]*np.sin(theta), war[0] * np.sin(theta) + war[1] * np.cos(theta)] for war in points]    
        return points

    def updatePrev(self, prevPoints, scale_factor, theta, translation_factor):     
        result = []

        for p in prevPoints:
            p = self.scaleX(p, scale_factor)
            p = self.rotateX(p, theta)
            p = np.add(p, translation_factor)
            result.append(p)
        return result
        




    def scalePoints(self, smaller, larger, a, b):

        #Distance calculations for small image vector and big image vector
        distance1 = np.sqrt(((smaller[a][0] - smaller[b][0]) ** 2 + (smaller[a][1] - smaller[b][1]) ** 2))
        distance2 = np.sqrt(((larger[a][0] - larger[b][0]) ** 2 + (larger[a][1] - larger[b][1]) ** 2))
        if(distance1 == 0):
            return None, None
        scale_factor = distance2/distance1
            
        
        smaller = np.multiply(smaller, scale_factor) #Scale points
        
        return smaller, scale_factor
        
        
    def rotatePoints(self, smaller, larger, a, b):    
        smaller = np.subtract(smaller, smaller[a]) #Translate smaller image points so best match is origin
        larger = np.subtract(larger, larger[a]) #Translate larger image points so best match is origin

        theta = np.arctan2(smaller[b][0], smaller[b][1]) - np.arctan2(larger[b][0], larger[b][1])
        smaller = [[war[0]*np.cos(theta) - war[1]*np.sin(theta), war[0] * np.sin(theta) + war[1] * np.cos(theta)] for war in smaller]    
        
        return smaller, theta



    def process_matches(self, smaller, larger, a, b):     
        smaller_copy = copy.deepcopy(smaller)
        
        #See this!!!: https://math.stackexchange.com/questions/1544147/find-transform-matrix-that-transforms-one-line-segment-to-another
        smaller_copy, scale_factor  = self.scalePoints(smaller_copy, larger, a, b)
        if(scale_factor == None):
            return [], None, None, None
        smaller_copy, rotate_factor = self.rotatePoints(smaller_copy, larger, a, b)
        smaller_copy, translation_factor = np.add(smaller_copy, larger[a]), larger[a] #translate back to original points
        
        return smaller_copy, scale_factor, rotate_factor, translation_factor



    def optimizePointSelection(self, kp1, kp2, matches):
        #Check for enough matches
        if len(matches) < 2:
            print("Error, not enough matches. Two Required, %d given\n" % len(matches))
            return []    
        
        print("Matches: %d\n" % len(matches))
        #Split matches into points from each image
        smaller = [] 
        larger = []
    
        for m in matches: #For loop which puts our matches as coordinate pairs into respective arrays
            smaller.append(kp1[m.queryIdx].pt)
            larger.append(kp2[m.trainIdx].pt)
            if (kp1[m.queryIdx].pt == None) ^ (kp2[m.trainIdx].pt == None):
                print(":( points don't all have corresponding point")
        
        
        #Choose reference points
        #These are the two indexes of points that form a line from each plot that are used for calculation
        
        errorList = [] #contains a list of tuples that contain the error and a tuple containing the endpoint indices for the line (ex: [(#, (a,b))])

        for a in range(len(matches)):
            for b in range(a, len(matches)): 
                if a == b:
                    continue
                temp_smaller, _, _, _ = self.process_matches(smaller, larger, a, b)
                if(len(temp_smaller) == 0):
                    continue
                errorList.append((self.calculateError(temp_smaller, larger), (a,b)))

        if(len(errorList) == 0):
            return []
        

        a_optimal, b_optimal = min(errorList)[1]

        
        smaller, scale_factor, rotate_factor, translation_factor = self.process_matches(smaller, larger, a_optimal, b_optimal)    
        
        return smaller, scale_factor, rotate_factor, translation_factor


    def findMatchesAndProcess(self, img2, img1):
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
        # CONSTANT_TEST Determines sensitivity of whats a good match
        # CONSTANT_TEST = 0.7 # <-- Default
        CONSTANT_TEST = 0.6 #Testing, .8 seems good. 

        good_matches = []
        for m,n in matches:
            if m.distance < CONSTANT_TEST*n.distance:
                good_matches.append(m)
        
        return self.optimizePointSelection(kp1, kp2, good_matches)


    # def showPointsOnImage(points, image):
    #     plt.imshow(image)
        
    #     x = [evil[0] for evil in points]
    #     y = [bastard[1] for bastard in points]
    #     plt.plot(x, y, 'ob')

    #     plt.show()
    #     plt.clf()


    def draw_points_on_image_cv(self, points, image):     
        new_img = np.zeros(image.shape, type(image.flat[0]))  
        new_img[0:image.shape[0],0:image.shape[1]] = image

        r = 10
        thickness = 2

        c = (0,255,0)

        for m in points:
            cv2.circle(new_img, tuple(np.round(m).astype(int)), r, c, thickness)

        return new_img

    def draw_all_points_on_image_cv(self, list_of_points, image):     
        new_img = np.zeros(image.shape, type(image.flat[0]))  
        new_img[0:image.shape[0],0:image.shape[1]] = image

        r = 3
        thickness = 1

        c = (0,255,0)

        for points in list_of_points:
            for m in points:
                cv2.circle(new_img, tuple(np.round(m).astype(int)), r, c, thickness)

        return new_img


####################################################################################################################
####################################################################################################################
####################################################################################################################

    cap = None
    writer = None
    # short_video = None


    def init(self):
        self.cap = cv2.VideoCapture(0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.writer = cv2.VideoWriter("flight video " + datetime.datetime.now().strftime("%I:%M:%S%p, %B %d, %Y") + ".mp4", cv2.VideoWriter_fourcc(*'MPEG'), 20, (width,height))

        # self.short_video = []



    def capture(self):
        ret,frame = self.cap.read()

        if ret == True:
            # cv2.imshow('test', frame)
            # cv2.waitKey(1)

            self.writer.write(frame)
            # self.short_video.append(frame)


    def closeWindows(self):        
        cv2.destroyAllWindows()



    def runAlgorithm(self):
        return
        # prev_img = self.short_video[0]

        # final_points = []

        # for frame in self.short_video[1:]:
            
        #     processed_points, scale_factor, rotate_factor, translation_factor = self.findMatchesAndProcess(frame, prev_img)
            
        #     if(len(processed_points) == 0):
        #         continue
            
        #     prevFinal = copy.deepcopy(final_points)

        #     final_points = self.updatePrev(final_points, scale_factor, rotate_factor, translation_factor)
        #     final_points.append(processed_points)
            

            

        #     cv2.imshow('test2', self.draw_points_on_image_cv(processed_points, frame))
        #     cv2.waitKey(50)

        #     cv2.imshow('All points', self.draw_all_points_on_image_cv(final_points, frame))
        #     cv2.waitKey(50)
            

        #     prev_img = frame
            
        # cv2.waitKey(0)


    def close(self):
        self.cap.release()
        self.writer.release()

        cv2.destroyAllWindows()

    """




    food
    1.
    fooy afjst od


    Take video/capture necessary images

    Match images to grid

    define matches in terms of coordinates w/ respect to grid
    (contingent on matching frame of video or captured image to the grid)

    Store the transformation for eah image successively

    Determine coordinates of the corners of last image with matches

    Apply matches until back to coordinates of original grid

    determine which grid square/s contain the last picture's corners






    Necessary from first match is the grid squares defined in terms of coordinates.

    Current Issues, needa filter the fisheye

    Track matches







    """

cap = cv2.VideoCapture(0)

# You should replace these 3 lines with the output in calibration step

DIM=(640, 480)
K=np.array([[369.02650618606265, 0.0, 328.86650517888114], [0.0, 369.26351345159935, 205.81189453142466], [0.0, 0.0, 1.0]])
D=np.array([[-0.04216729290605711], [0.0019285055730671665], [0.0053375517082646755], [-0.012079724576910731]])

def undistort(img):
    # img = cv2.imread(img_path)
    h,w = np.shape(img)[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img

cam = Camera()

cam.init()


prev_img = undistort(cap.read()[1])

final_points = []

while True:
    ret, frame = cap.read()

    if ret == True:
        frame = undistort(frame)
        processed_points, scale_factor, rotate_factor, translation_factor = cam.findMatchesAndProcess(frame, prev_img)
        
        if(len(processed_points) == 0):
            continue
        
        prevFinal = copy.deepcopy(final_points)

        final_points = cam.updatePrev(final_points, scale_factor, rotate_factor, translation_factor)
        final_points.append(processed_points)
        print(final_points)

        

        cv2.imshow('test2', cam.draw_points_on_image_cv(processed_points, frame))
        cv2.waitKey(50)

        cv2.imshow('All points', cam.draw_all_points_on_image_cv(final_points, frame))
        cv2.waitKey(50)
        

        prev_img = frame
    
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()