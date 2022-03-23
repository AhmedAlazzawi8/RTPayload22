import cv2
import datetime
import numpy

cap = cv2.VideoCapture(0)

while True:
	
	ret,frame = cap.read()
	
	if ret == True:
		

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == 27:
			cv2.imwrite("image capture " + datetime.datetime.now().strftime("%I:%M:%S%p, %B %d, %Y") + ".jpg",frame)



		if cv2.waitKey(1) & 0xFF == 81:
			break
	else: 
		break


cap.release()

cv2.destroyAllWindows()

