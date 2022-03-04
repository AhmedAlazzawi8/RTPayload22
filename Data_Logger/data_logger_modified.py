import smbus			#import SMBus module of I2C
import logging
import sys
import os
import board
from time import sleep          #import
from datetime import datetime
import adafruit_bmp3xx
import adafruit_bno055

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
		self.file = open("data.csv", "a")
		if os.stat("data.csv").st_size == 0:
				self.file.write("Time,Pressure,Temperature,Altitude,AccX,AccY,AccZ,magX,magY,magZ,GyroX,GyroY,GyroZ,roll,pitch,yaw"+
				"qX,qY,qZ,linAccX,linAccY,linAccZ,GravX,GravY,GravZ\n")

		self.bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
		self.Device_Address = 0x68   # MPU6050 device address
		
		# self.MPU_Init()
		print (" Reading Data of Gyroscope and Accelerometer")

		# I2C setup for BMP
		i2c = board.I2C()  # uses board.SCL and board.SDA
		self.bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

		self.bmp.pressure_oversampling = 8
		self.bmp.temperature_oversampling = 2
		self.bmp.sea_level_pressure = 1013.25

		i2c = board.I2C()
		self.sensor = adafruit_bno055.BNO055_I2C(i2c)
		# last_val = 0xFFFF

		#function to call data collection loop
		#data_Collection()

	def data_Collection(self):
		datamessage = str(datetime.now())+","

		print("Accelerometer (m/s^2): {}".format(self.sensor.acceleration))
		print("Magnetometer (microteslas): {}".format(self.sensor.magnetic))
		print("Gyroscope (rad/sec): {}".format(self.sensor.gyro))
		print("Euler angle: {}".format(self.sensor.euler))
		print("Quaternion: {}".format(self.sensor.quaternion))
		print("Linear acceleration (m/s^2): {}".format(self.sensor.linear_acceleration))
		print("Gravity (m/s^2): {}".format(self.sensor.gravity))
		
		datamessage = datamessage + str(self.sensor.acceleration[0]) + "," + str(self.sensor.acceleration[1]) + "," + str(self.sensor.acceleration[2]) + ","
		datamessage = datamessage + str(self.sensor.magnetic[0]) + "," + str(self.sensor.magnetic[1]) + "," + str(self.sensor.magnetic[2]) + ","
		datamessage = datamessage + str(self.sensor.gyro[0]) + "," + str(self.sensor.gyro[1]) + "," + str(self.sensor.gyro[2]) + ","
		datamessage = datamessage + str(self.sensor.euler[0]) + "," + str(self.sensor.euler[1]) + "," + str(self.sensor.euler[2]) + ","
		datamessage = datamessage + str(self.sensor.quaternion[0]) + "," + str(self.sensor.quaternion[1]) + "," + str(self.sensor.quaternion[2]) + ","
		datamessage = datamessage + str(self.sensor.linear_acceleration[0]) + "," + str(self.sensor.linear_acceleration[1]) + "," + str(self.sensor.linear_acceleration[2]) + ","
		datamessage = datamessage + str(self.sensor.gravity[0]) + "," + str(self.sensor.gravity[1]) + "," + str(self.sensor.gravity[2]) + ","
		
		print("Pressure: {:6.4f}  Temperature: {:5.2f} Altitude: {} meters".format(self.bmp.pressure, self.bmp.temperature, self.bmp.altitude))
		datamessage = datamessage + str(self.bmp.pressure) +"," + str(self.bmp.temperature) +"," + str(self.bmp.altitude) + "\n"

		self.file.write(datamessage)
		self.file.flush()
		
	def estimate(self):
		## DATA_LOAD ##
		
		AccX_Value = np.asarray(AccX)
		size = len(AccX_Value)
		textfile.close()
		
		## SET PARAMETERS ##
		dt = 0.01
		AccX_Variance = 0.0020
		
		# F Transmission Matrix
		F = [[1, dt, 0.*dt**2],
			 [0, 1,        dt],
			 [0, 0,         1]]
			 
		# Observatin Matrix
		H = [0,0,1]
		
		# inital_state_mean
		X0 = [0,
			  0,
			  AccX_Value[0]]
			  
		# initial_state_covariance 
		P0 = [[0, 0,             0],
			  [0, 0,             0],
			  [0, 0, AccX_Variance]]
			  
		n.timesteps = Accx_Value.shape[0]
		n_dim_state = 3
		filtered_state_means = np.zeros((n_timesteps, n_dim_state))
		filtered_state_covariance = np.zeroes((n_timesteps, n_dim_state, n_dim_state))
		
		kf = Kalmanfilter(transition_matricies = F,
						  observation_matricies = H,
						  transition_covariance = Q,
						  observation_covariance = R,
						  initial_state_means = X0,
						  initial_state_covariance = P0)
		
		# Iterative estimation for each new measurement
		for t in range(n_timesteps):
			if t == 0:
				filtered_state_measn[t] = X0;
				filtered_state_covariance = P0
			else:
				filtered_state_means[t], filtered_state_covariance[t] = {
					kf.filter_update(
						filter_state_means[t-1],
						filter_state_covariance[t-1],
						AccX_value[t]
					)
				)

	def close_file(self):
		self.file.close()

