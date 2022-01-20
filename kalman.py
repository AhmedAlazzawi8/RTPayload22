import smbus			#import SMBus module of I2C
import logging
import sys
import os
import board
from time import sleep          #import
from datetime import datetime
import adafruit_bmp3xx
import adafruit_bno055

import numpy as np
import pykalman as KalmanFilter

###############################
# SENSOR INITIALISATION
###############################

#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

#File for logging
file = open("/data.csv", "a")
if os.stat("/data.csv").st_size == 0:
        file.write("Time,Gx,Gy,Gz,Ax,Ay,Az,Pressure,Temperature,Altitude,AccX,AccY,AccZ,magX,magY,magZ,GyroX,GyroY,GyroZ,roll,pitch,yaw"+
		"qX,qY,qZ,linAccX,linAccY,linAccZ,GravX,GravY,GravZ\n")
def MPU_Init():
	#write to sample rate register
	bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
	
	#Write to power management register
	bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
	
	#Write to Configuration register
	bus.write_byte_data(Device_Address, CONFIG, 0)
	
	#Write to Gyro configuration register
	bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
	
	#Write to interrupt enable register
	bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
	#Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
	
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address

MPU_Init()
print (" Reading Data of Gyroscope and Accelerometer")

# I2C setup for BMP
i2c = board.I2C()  # uses board.SCL and board.SDA
bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

bmp.pressure_oversampling = 8
bmp.temperature_oversampling = 2
bmp.sea_level_pressure = 1013.25

i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)
last_val = 0xFFFF

############################################
# KALMAN FILTER MATRICIES
############################################

dt = 0.01

# 3 axis transition matrix:
F = [[1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0], 
     [0,  1,       dt,  0, 0, 0, 0, 0, 0],
     [0,  0,        1,  0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0, 0], 
     [0, 0, 0, 0,  1,        dt, 0, 0, 0],
     [0, 0, 0, 0,  0,         1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2], 
     [0, 0, 0, 0, 0, 0, 0,  1,        dt],
     [0, 0, 0, 0, 0, 0, 0,  0,         1]] # 9x9

# 3 axis transition covariance (starting with identiy matrix), this is a tuning parameter and will require some trial and error to get right
Q = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1]]

# 3 axis observation covariance (staring with identity matrix), this represents the standard deviations of sensors so update as necessary
# we can get an estimate by recording sensor data for a long period of time while the sensors are stationary and undisturbed then get the avg and standard deviation values
R_acc_X = 1 # covariance of x axis axis measurement
R_acc_Y = 1 # covariance of y axis axis measurement
R_acc_Z = 1 # covariance of z axis axis measurement

R = [[R_acc_X, 0, 0],
     [0, R_acc_Y, 0],
     [0, 0, R_acc_Z]]

# 3 axis observation matrix
H = [0, 0, 1, 0, 0, 1, 0, 0, 1] 


# 3 axis state matrix
X = [
  		0, # px
  		0, # vx
  		1, # ax Replace with initial reading
  		0, # py
  		0, # vy
  		1, # ay Replace with initial reading
  		0, # pz
  		0, # vz
  		1  # az Replace with initial reading
		]

P = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, R_acc_X, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, R_acc_Y, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, R_acc_Z]]


# initialize KF object
kf = KalmanFilter(transition_matrices = F, 
                  observation_matrices = H, 
                  transition_covariance = Q, 
                  observation_covariance = R, 
                  initial_state_mean = X, 
                  initial_state_covariance = P)

# make filtered arrays
n_dim_state = 3
filtered_state_means = np.zeros((n_dim_state))
filtered_state_covariances = np.zeros((n_dim_state, n_dim_state))
# filtered_state_means = X
# filtered_state_covariances = P

while True:
	##### READ DATA #####
    
	#Read Accelerometer raw value
	acc_x = read_raw_data(ACCEL_XOUT_H)
	acc_y = read_raw_data(ACCEL_YOUT_H)
	acc_z = read_raw_data(ACCEL_ZOUT_H)
	
	#Read Gyroscope raw value
	gyro_x = read_raw_data(GYRO_XOUT_H)
	gyro_y = read_raw_data(GYRO_YOUT_H)
	gyro_z = read_raw_data(GYRO_ZOUT_H)
	
	#Full scale range +/- 250 degree/C as per sensitivity scale factor
	Ax = acc_x/16384.0
	Ay = acc_y/16384.0
	Az = acc_z/16384.0
	
	Gx = gyro_x/131.0
	Gy = gyro_y/131.0
	Gz = gyro_z/131.0
	
	# BMP Data
	temp = bmp.temperature
	pres = bmp.pressure
	alt = bmp.altitude
	
	# BNO Data
	AccX = sensor.acceleration[0]
	AccY = sensor.acceleration[1]
	AccZ = sensor.acceleration[2]
	
	GyroX = sensor.gyro[0]
	GyroY = sensor.gyro[1]
	GyroZ = sensor.gyro[2]
	
	MagX = sensor.magnetic[0]
	MagY = sensor.magnetic[1]
	Magz = sensor.magnetic[2]
	
	euleX = sensor.euler[0]
	euleY = sensor.euler[1]
	euleZ = sensor.euler[2]
	
	LinAccX = sensor.linear_acceleration[0]
	LinAccY = sensor.linear_acceleration[1]
	LinAccZ = sensor.linear_acceleration[2]
	
	QuatX = sensor.quaternion[0]
	QuatY = sensor.quaternion[1]
	QuatZ = sensor.quaternion[2]
	
	GravX = sensor.gravity[0]
	GravY = sensor.gravity[1]
	GravZ = sensor.gravity[2]
	
    ##### KALMAN FILTER #####
    if initial iteration:
        filtered_state_means = X
        filtered_state_covariances = P
    else: 
        # update kf
        filtered_state_means, filtered_state_covariances = (
        kf.filter_update(
          filtered_state_means,
          filtered_state_covariances,
        )
	sleep(0.25)

file.close()