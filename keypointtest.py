import numpy as np
import cv2
from matplotlib import pyplot as plt

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
    
    
    
#     for m in matches:
#         cv2.circle(new_img, tuple(np.round(kp2[m.trainIdx].pt).astype(int)), r, c, thickness)

    cv2.circle(new_img, tuple(np.round(kp2[matches[0].trainIdx].pt).astype(int)), r, c, thickness)
    cv2.circle(new_img, tuple(np.round(kp2[matches[1].trainIdx].pt).astype(int)), r, c, thickness)


    return new_img

"""

"""
def scalePoints(kp1, kp2, matches):
    """Scales the provided points given the matches and keypoints of the two images and returns a new set of points
        The closer image (subimage) is number 1, the further is 2
    """
    
    
    
    if len(matches) < 2:
        print("Error, not enough matches. Two Required, %d given\n" % len(matches))
        raise "ERROR"
        return None
    
    
    smaller = [] 
    larger = []
   
    for m in matches: #For loop which puts our matches as coordinate pairs into respective arrays
        smaller.append(kp1[m.queryIdx].pt)
        larger.append(kp2[m.trainIdx].pt)
    
    #See this!!!: https://math.stackexchange.com/questions/1544147/find-transform-matrix-that-transforms-one-line-segment-to-another

#     smaller = [[0,0], [1,0], [2, 0], [2,2]]
#     larger = [[1,1], [1,3], [1, 5], [-3,5]]
    
    
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')
    
    plt.show()
    plt.clf()
    
    #----------------------------------------Scaling---------------------------------------
    smlPt_0 = smaller[0] # X, Y Coords of our best match on the smaller image
    smlPt_1 = smaller[1] # X, Y coords of our second best match on smaller image
    
    bigPt_0 = larger[0] # X, Y Coords of our best match on the larger image
    bigPt_1 = larger[1] # X, Y Coords of our second best match on the larger image
    
    
    #Distance calculations for small image vector and big image vector
    distance1 = np.sqrt(((smlPt_0[0] - smlPt_1[0]) ** 2 + (smlPt_0[1] - smlPt_1[1]) ** 2))
    distance2 = np.sqrt(((bigPt_0[0] - bigPt_1[0]) ** 2 + (bigPt_0[1] - bigPt_1[1]) ** 2))
    
    scale_factor = distance2/distance1
        
    
    smaller = np.subtract(smaller, smaller[0]) #Translate smaller image points so best match is origin
    larger = np.subtract(larger, larger[0]) #Translate larger image points so best match is origin
    
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')
    
    plt.show()
    plt.clf()
    
    smaller = np.multiply(smaller, scale_factor) #Scale points
    
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')
#     
#     print(larger)
#     print(smaller)
#     
    plt.show()
    plt.clf()
    #Rotation
    
    
    
    theta = np.arctan2([larger[1][1], larger[0][1], smaller[1][1], smaller[0][1]], [larger[1][0], larger[0][0], smaller[1][0], smaller[0][0]])[0]
    #Rotation Matrix
    #R = np.array([np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
    
    print(theta)
    #print([smaller[1][0]*np.cos(theta[0]) - smaller[1][1]*np.sin(theta[0]), smaller[1][0] * np.sin(theta[0]) + smaller[1][0] * np.cos(theta[0])])
    smaller = [[war[0]*np.cos(theta) - war[1]*np.sin(theta), war[0] * np.sin(theta) + war[1] * np.cos(theta)] for war in smaller]
    #smaller = np.matmul(smaller, R)
    
    print(larger)
    print(smaller)
    x = [evil[0] for evil in smaller]
    y = [bastard[1] for bastard in smaller]
    plt.plot(x, y, 'ob')
    x = [evil[0] for evil in larger]
    y = [bastard[1] for bastard in larger]
    plt.plot(x, y, '*r')
    
    plt.show()
    
    
    
    

    #------------------------------------------Rotate-------------------------------------
    #(1x2)(2x2) =(1x2) img1pts * matrix = bigimgpts
    
    #move the big image to origin to calculate rotation properly
    #big_at_origin = np.subtract()
    
    #theta = arccos(dot product/(size**2))
    
    #Calculate Rotation angle
    
    
    #Apply rotation
    
    #-----------------------------------------Translation Two-----------------------------------
    
    
    
    
    
    
    return smaller



def process_matches(img1, kp1, img2, kp2, matches):  
    points = []
    
    points = scalePoints(kp1, kp2, matches)
    
    return points



MIN_MATCH_COUNT = 1

img2 = cv2.imread('pic1.jpg')      # queryImage
img1 = cv2.imread('pic1_closer.jpg')             # trainImage

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
CONSTANT_TEST = 0.6 #Testing, 8 seems good. 

good = []
for m,n in matches:
    if m.distance < CONSTANT_TEST*n.distance:
        good.append(m)

print("There are %d good matches" % len(good))
# if len(good)>MIN_MATCH_COUNT:
#     # ERROR with this code vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv This is what was causing the random line
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = img1.shape[0], img1.shape[1]                  #This line was causing problems, cause the image can be len 3. I modified to fix it
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)

#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#     

# else:
#     print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#     # print( "Not enough matches are found")
#     matchesMask = None
    
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
process_matches(img1, kp1, img2, kp2, good)