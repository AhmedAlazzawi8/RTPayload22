import cv2

cv2.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
capture = cv2.CaptureFromCAM(0)

def repeat():
  frame = cv2.QueryFrame(capture)
  cv2.ShowImage("w1", frame)


while True:
  repeat()
