import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# You should replace these 3 lines with the output in calibration step

DIM=(640, 480)
K=np.array([[369.02650618606265, 0.0, 328.86650517888114], [0.0, 369.26351345159935, 205.81189453142466], [0.0, 0.0, 1.0]])
D=np.array([[-0.04216729290605711], [0.0019285055730671665], [0.0053375517082646755], [-0.012079724576910731]])

def undistort(img):
    # img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img

if __name__ == "__main__":
    while True:
        ret,frame = cap.read()

        if ret == True:

            cv2.imshow('distorted', frame)

            cv2.imshow("undistorted", undistort(frame))

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else: 
            break


    cap.release()
    cv2.destroyAllWindows()
