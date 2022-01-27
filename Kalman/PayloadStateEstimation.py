# Payload state estimation algorithm
# libraries: filterpy, pykalman
# pykalman example: https://coderedirect.com/questions/257834/using-pykalman-on-raw-acceleration-data-to-calculate-position


### CDR algorithm ###

# Upload launch site info (site size, cell size, estimated starting position)
#
# Initialize:
# 	- init sensors and libraries
# 	- init sensor values and arrays
#   - init timestep and freq
# 	- create KF matrices:
#			- F // transition mat
#     - H // observation 
#     - Q // transition covariance 
#     - R // observation covariance 
#     - X // state
#     - P // state covariance
#		- init kf object


# Run: 
# 	While (not landed) do (at each time step):
#			- Read sensor data
#			- Update sensor storage arrays/files
#		End
#		Update KF
# 	Convert displacement to cells
#		Return cells


# algorithm starts here:

# import sensor libraries
#

# import kalman filter libraries
import numpy as np
from pykalman import KalmanFilter

# init sensors

# create matrices:

# simple 1 axis transtion matrix:
F = [[1, dt, 0.5*dt**2], 
     [0,  1,       dt],
     [0,  0,        1]]

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

# live implementation
while some condition:
  if initial iteration:
    filtered_state_means = X
    filtered_state_covariances = P
  else: 
    if a new timestep has occured:
      # read new sensor data
      accX = # read x accel from bno
      accY = # read y accel from bno
      accZ = # read z accel from bno
      # update kf
      filtered_state_means, filtered_state_covariances = (
      kf.filter_update(
          filtered_state_means,
          filtered_state_covariances,
          # bno data
      )
        
      # we can print or send data with radio here
        

# convert displacement to grid square
    
        
# algorithm end
  



### from pykalman example, used for refrence ###
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

load_data()

# Data description
#  Time
#  AccX_HP - high precision acceleration signal
#  AccX_LP - low precision acceleration signal
#  RefPosX - real position (ground truth)
#  RefVelX - real velocity (ground truth)

# switch between two acceleration signals
use_HP_signal = 1

if use_HP_signal:
    AccX_Value = AccX_HP
    AccX_Variance = 0.0007
else:    
    AccX_Value = AccX_LP
    AccX_Variance = 0.0020


# time step
dt = 0.01

# transition_matrix  
F = [[1, dt, 0.5*dt**2], 
     [0,  1,       dt],
     [0,  0,        1]]

# observation_matrix   
H = [0, 0, 1]

# transition_covariance 
Q = [[0.2,    0,      0], 
     [  0,  0.1,      0],
     [  0,    0,  10e-4]]

# observation_covariance 
R = AccX_Variance

# initial_state_mean
X0 = [0,
      0,
      AccX_Value[0, 0]]

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
            AccX_Value[t, 0]
        )
    )
