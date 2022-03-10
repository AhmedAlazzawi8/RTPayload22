import sys
import uptime
# import threading

sys.path.insert(0, 'Data_Logger')
import data_logger_modified as DataLogger

sys.path.insert(0, 'Camera Processing')
import keypointtest as Camera
from time import sleep

def curr_millis():
    return uptime.uptime() * 1000


run = True
log = DataLogger.Logger()
cam = Camera.Camera()


def dataMagic():
    while(run):
        log.data_Collection()

def camCap():
    while(run):
        cam.capture()


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

# threading.Thread.start("data logging", dataMagic())
# threading.Thread.start("camera", camCap())
state = 0
alt_prv = log.altitude()
while(True):
    if (state == 1 and run):
    
        if curr_millis() - last_data_time >= 1000/data_hz: 
            data_time += curr_millis() - last_data_time
            data_count += 1
            log.data_Collection()
            last_data_time = curr_millis()
            
        if curr_millis() - last_camera_time >= 1000/camera_hz: 
            cam_time += curr_millis() - last_camera_time
            cam_count += 1
            cam.capture()
            last_camera_time = curr_millis()
        
        # if land_condition:
        if (curr_millis() - entry_time) >= 350000:
            run = False

    elif (state ==0 and run):
        alt_curr = log.altitude()
        diff = alt_prv - alt_curr
        if(diff > 0.7  or diff < -0.7):
            print("Launch!")
            entry_time = curr_millis()
            state = 1
        else: 
            state = 0
        alt_prv = alt_curr
        sleep(1)

    elif(not run):
        break


cam.closeWindows()
log.close_file()
#print("Data Frequency = ", 1000*data_count/data_time)
#print("Camera Frequency = ", 1000*cam_count/cam_time)


#Process data
#cam.runAlgorithm()
print("Landed")
location = log.estimate(1/(1000*data_count/data_time))
print(location)

# Send location
# Radio.send_result(location)


# Clean up
cam.close()
log.close_file()
