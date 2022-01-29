from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

### LOAD DATA HERE ###
text_file = open("/Users/khushishah/Downloads/vertialacceldata.txt", "r")
lines = text_file.read().split('\n')
for i in range(0, len(lines)):
    lines[i] = float(lines[i])
AccX_Value = np.asarray(lines)

print(AccX_Value)
print(len(AccX_Value))
text_file.close()
# Data description
#  Time
#  AccX_HP - high precision acceleration signal
#  AccX_LP - low precision acceleration signal
#  RefPosX - real position (ground truth)
#  RefVelX - real velocity (ground truth)
AccX_Variance = 0.0020


# time step
#dt = 15.93/len(AccX_Value)
dt = 0.1
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


f, axarr = plt.subplots(3, sharex=True)

#deltat = 15.93/len(AccX_Value)
Time = np.arange(0, 176, 1)
RefVelX = [0]
# v = v0 + at
for i in range(1, len(AccX_Value)):
    v = RefVelX[i-1] + AccX_Value[i-1]*dt
    RefVelX.append(v)

RefPosX = [0]
# x = x0 + vt + 1/2at^2
for i in range(1, len(AccX_Value)):
    x = RefPosX[i-1] + RefVelX[i-1]*dt + 0.5*AccX_Value[i-1]*dt
    RefPosX.append(x)

axarr[0].plot(Time, AccX_Value, label="Input AccX")
axarr[0].plot(Time, filtered_state_means[:, 2], "y-", label="Estimated AccX")
axarr[0].plot(Time, filtered_state_means[:, 2]-AccX_Value, "r-", label="Error")
axarr[0].set_title('Acceleration X')
axarr[0].grid()
axarr[0].legend()
#axarr[0].set_ylim([-4, 4])

axarr[1].plot(Time, RefVelX, label="Reference VelX")
axarr[1].plot(Time, filtered_state_means[:, 1], "y-", label="Estimated VelX")
axarr[1].plot(Time, filtered_state_means[:, 1]-RefVelX, "r-", label="Error")
axarr[1].set_title('Velocity X')
axarr[1].grid()
axarr[1].legend()
#axarr[1].set_ylim([-1, 20])

axarr[2].plot(Time, RefPosX, label="Reference PosX")
axarr[2].plot(Time, filtered_state_means[:, 0], "y-", label="Estimated PosX")
axarr[2].plot(Time, filtered_state_means[:, 0]-RefPosX, "r-", label="Error")
axarr[2].set_title('Position X')
axarr[2].grid()
axarr[2].legend()
#axarr[2].set_ylim([-10, 1000])

plt.show()