import sys
import uptime
import cv2
import os
from datetime import datetime
from matcher import recordVideo
import RPi.GPIO as GPIO 
sys.path.insert(0, 'Data_Logger')
import data_logger_modified as DataLogger

from time import sleep

def curr_millis():
    return uptime.uptime() * 1000


run = True
log = DataLogger.Logger()



def dataMagic():
    while(run):
        log.data_Collection()




# Initialize class variables



temp_testing = 0

#These are upper thresholds. If it can't attain these frequencies, it can only go as fast as it can process
data_hz = 100
camera_hz = 60

last_data_time = curr_millis()
last_camera_time = curr_millis()

data_time = 0
data_count = 0
cam_time = 0
cam_count = 0
#Run loop
entry_time = 0

state = 0



if __name__ == "__main__":
    alt_prv = log.altitude()
    entry_time = curr_millis()
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(datetime.now()) + "flightVideo" + ".mp4", cv2.VideoWriter_fourcc(*'MPEG'), 20, (width,height))
    frameCounter = 0
    frameTime = open("frameTimes" + str(datetime.now()) + ".csv", "a")


    # setup log button
    LOG_PIN = 21
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LOG_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


    # wait while button is not pressed which results in a high signal being read since this is a pull-up config
    while(GPIO.input(LOG_PIN)):
        continue 


    while(not GPIO.input(LOG_PIN)):

        currentMill = curr_millis()
        if currentMill - last_data_time >= 10: 
            # print("logged data:", currentMill)
            # data_time += currentMill - last_data_time
            # data_count += 1
            log.data_Collection()
            last_data_time = currentMill
        
        
        ret, frame = cap.read()           
        if ret == True:
            frameTime.write("TimeStamp: " + str(datetime.now()) + " Framecounter: " + str(frameCounter) + "\n")
            frameTime.flush()
            writer.write(frame)
            frameCounter += 1
            


        # # if land_condition:
        # if (currentMill - entry_time) >= 100000: #Change back to 350k eventually
            
        #     run = False
        # else:
        #     print(currentMill - entry_time)



        # currentMill = curr_millis()
        # #print(currentMill)
        # if (state == 1 and run):
        #     #print("run")
        #     if currentMill - last_data_time >= 1000: 
        #         #print("logged data:", currentMill)
        #         data_time += currentMill - last_data_time
        #         data_count += 1
        #         log.data_Collection()
        #         last_data_time = currentMill
            
            
        #     ret, frame = cap.read()
            
            
        #     if ret == True:
        #         frameTime.write("TimeStamp: " + str(datetime.now()) + " Framecounter: " + str(frameCounter) + "\n")
        #         frameTime.flush()
        #         writer.write(frame)
        #         frameCounter += 1
                
        #     # if land_condition:
        #     if (currentMill - entry_time) >= 100000: #Change back to 350k eventually
                
               
        #         run = False
        #     else:
        #         print(currentMill - entry_time)

        # elif (state ==0 and run):
        #     alt_curr = log.altitude()
        #     diff = alt_prv - alt_curr
        #     state = 1 #THIS IS NECESSARY THE IF STATEMENT IS DUMB
        #     if(diff > 0.7  or diff < -0.7):
        #         print("Launch!")
        #         entry_time = curr_millis()
        #         state = 1
        #     #else: 
        #         #state = 0
        #     alt_prv = alt_curr
        #     #alt_prv = 0
        #     sleep(1)

        # elif(not run):
        #     break


    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    frameTime.close()
    log.close_file()
    #print("Data Frequency = ", 1000*data_count/data_time)
    #print("Camera Frequency = ", 1000*cam_count/cam_time)


    #Process data
    #cam.runAlgorithm()
    # print("Landed")
    #location = log.estimate(1/(1000*data_count/data_time))
    # print("E13 ")

    # Send location
    #Radio.send_result(location)


    # Clean up
    #log.close_file()
