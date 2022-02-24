import cv2
import numpy

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('testvideo.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 20, (width,height))


while True:
    ret,frame = cap.read()

    if ret == True:
        writer.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else: 
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
