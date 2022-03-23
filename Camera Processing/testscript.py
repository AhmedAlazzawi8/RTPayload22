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
    img = undistort(cap.read()[1])

    cv2.imshow('img', img)
    cv2.waitKey(50)