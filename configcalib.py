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

# I2C setup for BMP
i2c = board.I2C()  # uses board.SCL and board.SDA
bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

bmp.pressure_oversampling = 8
bmp.temperature_oversampling = 2
bmp.sea_level_pressure = 1013.25

#I2C setup for BNO
sensor = adafruit_bno055.BNO055_I2C(i2c)
last_val = 0xFFFF

########################
#    STD FUNCTION      #
########################

def get_sum(ls):
    s = 0
    for x in ls:
        if x != None:
            s += x
    return s

def get_std_dev(ls):
    n = len(ls)
    mean = get_sum(ls) / n
    var = 0
    for x in ls:
        if x != None:
            var += (x-mean)**2
        else:
            var += (0-mean)**2
    var = var/n    
    std_dev = var ** 0.5
    return std_dev

# dynamic Python Lists to store the values:
Ax = []
Ay = []
Az = []

Gx = []
Gy = []
Gz = []

temp = []
press= []
alt = []

AccX = []
AccY = []
AccZ = []

GyroX = []
GyroY = []
GyroZ = []

MagX = []
MagY = []
MagZ = []

roll = []
pitch = []
yaw = []

QuatX = []
QuatY = []
QuatZ = []

LinAccX = []
LinAccY = []
LinAccZ = []

GravX = []
GravY = []
GravZ = []

def main ():
    t = 0
    while t <= 25750:
        if t%429 == 0:
            print("Time: " + str(t/429) + " mins")
        
        #collect data in 30 arrays
        Ax.append(read_raw_data(ACCEL_XOUT_H) / 16384.0)
        Ay.append(read_raw_data(ACCEL_YOUT_H) / 16384.0)
        Az.append(read_raw_data(ACCEL_ZOUT_H) / 16384.0)
        
        Gx.append(read_raw_data(GYRO_XOUT_H)/131.0)
        Gy.append(read_raw_data(GYRO_YOUT_H)/131.0)
        Gz.append(read_raw_data(GYRO_ZOUT_H)/131.0)

        temp.append(bmp.temperature)
        alt.append(bmp.altitude)
        press.append(bmp.pressure)
        
        AccX.append(sensor.acceleration[0])
        AccY.append(sensor.acceleration[1])
        AccZ.append(sensor.acceleration[2])
        GyroX.append(sensor.gyro[0])
        GyroY.append(sensor.gyro[1])
        GyroZ.append(sensor.gyro[2])
        MagX.append(sensor.magnetic[0])
        MagY.append(sensor.magnetic[1])
        MagZ.append(sensor.magnetic[2])
        
        roll.append(sensor.euler[0])
        pitch.append(sensor.euler[1])
        yaw.append(sensor.euler[2])
        QuatX.append(sensor.quaternion[0])
        QuatY.append(sensor.quaternion[1])
        QuatZ.append(sensor.quaternion[2])
        LinAccX.append(sensor.linear_acceleration[0])
        LinAccY.append(sensor.linear_acceleration[1])
        LinAccZ.append(sensor.linear_acceleration[2])
        GravX.append(sensor.gravity[0])
        GravY.append(sensor.gravity[1])
        GravZ.append(sensor.gravity[2])
        
        t += 1
        sleep(0.02)
        
    stdAx = get_std_dev(Ax)
    stdAy = get_std_dev(Ay)
    stdAz = get_std_dev(Az)
    
    stdGx = get_std_dev(Gx)
    stdGy = get_std_dev(Gy)
    stdGz = get_std_dev(Gz)
    
    stdAccX = get_std_dev(AccX)
    stdAccY = get_std_dev(AccY)
    stdAccZ = get_std_dev(AccZ)
    stdGyroX = get_std_dev(GyroX)
    stdGyroY = get_std_dev(GyroY)
    stdGyroZ = get_std_dev(GyroZ)
    stdMagX = get_std_dev(MagX)
    stdMagY = get_std_dev(MagY)
    stdMagZ = get_std_dev(MagZ)
    
    stdRoll = get_std_dev(roll)
    stdPitch = get_std_dev(pitch)
    stdYaw = get_std_dev(yaw)
    stdQuatX = get_std_dev(QuatX)
    stdQuatY = get_std_dev(QuatY)
    stdQuatZ = get_std_dev(QuatZ)
    stdLinAccX = get_std_dev(LinAccX)
    stdLinAccY = get_std_dev(LinAccY)
    stdLinAccZ = get_std_dev(LinAccZ)
    stdGravX = get_std_dev(GravX)
    stdGravY = get_std_dev(GravY)
    stdGravZ = get_std_dev(GravZ)
    
    print()
    print("Standard Deviations:")
    print("Ax: " + str(stdAx))
    print("Ay: " + str(stdAy))
    print("Az: " + str(stdAz))
    print("Gx: " + str(stdGx))
    print("Gy: " + str(stdGy))
    print("Gz: " + str(stdGz))
    
    print("AccX: " + str(stdAccX))
    print("AccY: " + str(stdAccY))
    print("AccZ: " + str(stdAccZ))
    print("GyroX: " + str(stdGyroX))
    print("GyroY: " + str(stdGyroY))
    print("GyroZ: " + str(stdGyroZ))
    print("MagX: " + str(stdMagX))
    print("MagY: " + str(stdMagY))
    print("MagZ: " + str(stdMagZ))
    
    print("Roll: " + str(stdRoll))
    print("Pitch: " + str(stdPitch))
    print("Yaw: " + str(stdYaw))
    print("LinAccX: " + str(stdLinAccX))
    print("LinAccY: " + str(stdLinAccY))
    print("LinAccZ: " + str(stdLinAccZ))
    print("QuatX: " + str(stdQuatX))
    print("QuatY: " + str(stdQuatY))
    print("QuatZ: " + str(stdQuatZ))
    print("GravX: " + str(stdGravX))
    print("GravY: " + str(stdGravY))
    print("GravZ: " + str(stdGravZ))
        
if __name__=="__main__":
    main()
