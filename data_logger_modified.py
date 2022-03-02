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

def init_Collection():
	#File for logging
	file = open("/data.csv", "a")
	if os.stat("/data.csv").st_size == 0:
    	    file.write("Time,Gx,Gy,Gz,Ax,Ay,Az,Pressure,Temperature,Altitude,AccX,AccY,AccZ,magX,magY,magZ,GyroX,GyroY,GyroZ,roll,pitch,yaw"+
			"qX,qY,qZ,linAccX,linAccY,linAccZ,GravX,GravY,GravZ\n")

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

	#function to call data collection loop
	data_Collection()

def data_Collection(file):
	datamessage = str(datetime.now())+","
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
	

	print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az) 	
	    # Read the Euler angles for heading, roll, pitch (all in degrees).
	datamessage = datamessage + str(Gx) +"," +str(Gy) +"," +str(Gz) +"," + str(Ax) +"," + str(Ay) +"," + str(Az) +","
	
	print("Pressure: {:6.4f}  Temperature: {:5.2f} Altitude: {} meters".format(bmp.pressure, bmp.temperature, bmp.altitude))
	datamessage = datamessage + str(bmp.pressure) +"," + str(bmp.temperature) +"," + str(bmp.altitude) + ","

	print("Accelerometer (m/s^2): {}".format(sensor.acceleration))
	print("Magnetometer (microteslas): {}".format(sensor.magnetic))
	print("Gyroscope (rad/sec): {}".format(sensor.gyro))
	print("Euler angle: {}".format(sensor.euler))
	print("Quaternion: {}".format(sensor.quaternion))
	print("Linear acceleration (m/s^2): {}".format(sensor.linear_acceleration))
	print("Gravity (m/s^2): {}".format(sensor.gravity))
	
	datamessage = datamessage + str(sensor.acceleration[0]) + "," + str(sensor.acceleration[1]) + "," + str(sensor.acceleration[2]) + ","
	datamessage = datamessage + str(sensor.magnetic[0]) + "," + str(sensor.magnetic[1]) + "," + str(sensor.magnetic[2]) + ","
	datamessage = datamessage + str(sensor.gyro[0]) + "," + str(sensor.gyro[1]) + "," + str(sensor.gyro[2]) + ","
	datamessage = datamessage + str(sensor.euler[0]) + "," + str(sensor.euler[1]) + "," + str(sensor.euler[2]) + ","
	datamessage = datamessage + str(sensor.quaternion[0]) + "," + str(sensor.quaternion[1]) + "," + str(sensor.quaternion[2]) + ","
	datamessage = datamessage + str(sensor.linear_acceleration[0]) + "," + str(sensor.linear_acceleration[1]) + "," + str(sensor.linear_acceleration[2]) + ","
	datamessage = datamessage + str(sensor.gravity[0]) + "," + str(sensor.gravity[1]) + "," + str(sensor.gravity[2]) + "\n"

	file.write(datamessage)
	file.flush()

def close_file(file):
	file.close()

