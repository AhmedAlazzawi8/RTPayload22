import sys

sys.path.insert(0, 'Data_Logger')
import data_logger.py
sys.path.insert(0, 'Camera Processing')
import keypointtest.py


# Initialize class variables
run = True

# Initialize all modules
Data_Logger_init()
Camera_init()


#Run loop
while(run):
    Data_Logger_tick()
    Camera_tick()

    # if landed:
    #     run = false

#Process data
Camera_Run_Algorithm()
Data_Logger_estimation()


# Send location
Radio_send_result(location)


# Clean up
Data_Logger_close()
