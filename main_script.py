import sys
import uptime

sys.path.insert(0, 'Data_Logger')
import data_logger_modified as DataLogger

# sys.path.insert(0, 'Camera Processing')
# import keypointtest as Camera

def curr_millis():
    return uptime.uptime() * 1000




# Initialize class variables
run = True
log = DataLogger.Logger()

# Initialize all modules
# Camera.init()


temp_testing = 0


data_hz = 100
camera_hz = 60

last_data_time = curr_millis()
last_camera_time = curr_millis()

#Run loop
while(run):
    if curr_millis() - last_data_time >= 1000/data_hz: 
        log.data_Collection()
        last_data_time = curr_millis()
        
    if curr_millis() - last_camera_time >= 1000/camera_hz: 
        # Camera.capture()
        last_camera_time = curr_millis()
    
    # if land_condition:
    if temp_testing >= 6000:
        run = False
    
    temp_testing += 1

# Camera.closeWindows()

#Process data
# Camera.runAlgorithm()
#log.estimate()


# Send location
# Radio.send_result(location)


# Clean up
# Camera.close()
log.close_file()
