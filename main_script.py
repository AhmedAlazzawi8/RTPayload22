import sys
import uptime
# import threading

sys.path.insert(0, 'Data_Logger')
import data_logger_modified as DataLogger

sys.path.insert(0, 'Camera Processing')
import keypointtest as Camera

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


data_hz = 100
camera_hz = 60

last_data_time = curr_millis()
last_camera_time = curr_millis()

data_time = 0
data_count = 0
cam_time = 0
cam_count = 0
#Run loop

# threading.Thread.start("data logging", dataMagic())
# threading.Thread.start("camera", camCap())
while(run):
    
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
    if temp_testing >= 500:
        run = False
    
    temp_testing += 1

cam.closeWindows()
#print("Data Frequency = ", 1000*data_count/data_time)
#print("Camera Frequency = ", 1000*cam_count/cam_time)


#Process data
#cam.runAlgorithm()
#log.estimate(1000*data_count/data_time)


# Send location
# Radio.send_result(location)


# Clean up
cam.close()
log.close_file()
