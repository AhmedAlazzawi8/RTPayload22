import smbus			#import SMBus module of I2C
import logging
import sys
import os
import board
from time import sleep          #import
from datetime import datetime
import adafruit_bmp3xx
import adafruit_bno055
import adafruit_gps
import numpy as np
from pykalman import KalmanFilter
import glob

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


class Logger:

	file = None
	bus = None
	sensor = None
	bmp = None
	Device_Address = None

	def __init__(self):
		self.file = None
		self.bus = None
		self.sensor = None
		self.bmp = None
		self.Device_Address = None
		self.init_Collection()

	def init_Collection(self):
		#File for logging
		self.file = open("data" + str(datetime.now()) + ".csv", "a")
		self.file.write("Time,Pressure,Temperature,Altitude,AccX,AccY,AccZ,magX,magY,magZ,GyroX,GyroY,GyroZ,roll,pitch,yaw"+
				"qX,qY,qZ,linAccX,linAccY,linAccZ,GravX,GravY,GravZ\n")

		self.bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
		self.Device_Address = 0x68   # MPU6050 device address
		
		# self.MPU_Init()
		#print (" Reading Data of Gyroscope and Accelerometer")

		# I2C setup for BMP
		i2c = board.I2C()  # uses board.SCL and board.SDA
		self.bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

		self.bmp.pressure_oversampling = 8
		self.bmp.temperature_oversampling = 2
		self.bmp.sea_level_pressure = 1013.25

		i2c = board.I2C()
		self.sensor = adafruit_bno055.BNO055_I2C(i2c)
		# last_val = 0xFFFF
		self.gps = adafruit_gps.GPS_GtopI2C(i2c)
		self.gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
		self.gps.send_command(b"PMTK220,1000")

		#function to call data collection loop
		#data_Collection()

	def data_Collection(self):
		datamessage = str(datetime.now())+","

		# print("Accelerometer (m/s^2): {}".format(self.sensor.acceleration))
		# print("Magnetometer (microteslas): {}".format(self.sensor.magnetic))
		# print("Gyroscope (rad/sec): {}".format(self.sensor.gyro))
		# print("Euler angle: {}".format(self.sensor.euler))
		# print("Quaternion: {}".format(self.sensor.quaternion))
		# print("Linear acceleration (m/s^2): {}".format(self.sensor.linear_acceleration))
		# print("Gravity (m/s^2): {}".format(self.sensor.gravity))
		
		datamessage = datamessage + str(self.sensor.acceleration[0]) + "," + str(self.sensor.acceleration[1]) + "," + str(self.sensor.acceleration[2]) + ","
		# datamessage = datamessage + str(self.sensor.magnetic[0]) + "," + str(self.sensor.magnetic[1]) + "," + str(self.sensor.magnetic[2]) + ","
		datamessage = datamessage + str(self.sensor.gyro[0]) + "," + str(self.sensor.gyro[1]) + "," + str(self.sensor.gyro[2]) + ","
		datamessage = datamessage + str(self.sensor.euler[0]) + "," + str(self.sensor.euler[1]) + "," + str(self.sensor.euler[2]) + ","
		datamessage = datamessage + str(self.sensor.quaternion[0]) + "," + str(self.sensor.quaternion[1]) + "," + str(self.sensor.quaternion[2]) + ","
		datamessage = datamessage + str(self.sensor.linear_acceleration[0]) + "," + str(self.sensor.linear_acceleration[1]) + "," + str(self.sensor.linear_acceleration[2]) + ","
		datamessage = datamessage + str(self.sensor.gravity[0]) + "," + str(self.sensor.gravity[1]) + "," + str(self.sensor.gravity[2]) + ","
		
		data = self.gps.read(32)
		if data is not None:
			data_str = "".join([chr(b) for b in data])
			datamessage = datamessage + data_str + ","
		else:
			datamessage = datamessage + "0" + ","

		# print("Pressure: {:6.4f}  Temperature: {:5.2f} Altitude: {} meters".format(self.bmp.pressure, self.bmp.temperature, self.bmp.altitude))
		datamessage = datamessage + str(self.bmp.pressure) +"," '''+ str(self.bmp.temperature) +","''' + str(self.bmp.altitude) + "\n"

		self.file.write(datamessage)
		self.file.flush()
	
	def kalman(self, dt, AccX_Value):
		AccX_Variance = 0.0020

		# transition_matrix
		F = [[1, dt, 0.5*dt**2],
			[0,  1,       dt],
			[0,  0,        1]]

		# observation_matrix   
		H = [0, 0, 1]

		# transition_covariance 
		Q = [[0.5,    0,      0],
			[  0,  0.2,      0],
			[  0,    0,    0.1]]

		# observation_covariance 
		R = AccX_Variance

		# initial_state_mean
		X0 = [0,
			0,
			AccX_Value[0]]

		# initial_state_covariance
		P0 = [[  0,    0,               0],
			[  0,    0,               0],
			[  0,    0,   AccX_Variance]]

		n_timesteps = AccX_Value.shape[0]
		n_dim_state = 3
		filtered_state_means = np.zeros((n_timesteps, n_dim_state))
		filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

		kf = KalmanFilter(transition_matrices = F,
						observation_matrices = H,
						transition_covariance = Q,
						observation_covariance = R,
						initial_state_mean = X0,
						initial_state_covariance = P0)

		# iterative estimation for each new measurement
		for t in range(n_timesteps):
			if t == 0:
				filtered_state_means[t] = X0
				filtered_state_covariances[t] = P0
			else:
				filtered_state_means[t], filtered_state_covariances[t] = (
					kf.filter_update(
						filtered_state_means[t-1],
						filtered_state_covariances[t-1],
						AccX_Value[t]
					)
				)

		lenX = len(filtered_state_means[:,0])
		Final_Pos = filtered_state_means[lenX-1, 0]
		return Final_Pos

	def estimate(self, dt):
		## DATA_LOAD ##
		#creates list of all files that have .csv extension
		file_list = glob.glob('*.csv') #finds all files in current directory with .csv extension
		latest_file = max(file_list)

		#print(latest_file)

		text_file = open(latest_file, "r")
		Acc = text_file.read().split('\n')
		AccX = []
		AccY = []

		for i in range(1, len(Acc)):
			Acc[i] = Acc[i].split(',')
			try:
				AccX.append(float(Acc[i][1]))
				AccY.append(float(Acc[i][2]))
			except:
				ValueError
		
		text_file.close()
		
		AccX_Value = np.asarray(AccX)
		AccY_Value = np.asarray(AccY)

		pos = []
		pos.append(self.kalman(dt, AccX_Value))
		pos.append(self.kalman(dt, AccY_Value))

		final = "X: " + str(pos[0]) + "\nY: " + str(pos[1]) + "\n"

		text_file = open(latest_file, "a")
		text_file.write(final)
		text_file.close()

		return pos

	def altitude(self):
		return self.bmp.altitude

	def close_file(self):
		self.file.close()

