import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def sfilter(array):
    b, a = signal.butter(2, 0.33/2.)
    y = signal.filtfilt(b, a, array)
    return y


data = np.loadtxt('./blood/log.txt')

time, xc, yc, a, b, theta0, XC, YC, R = data.T
time *= 0.33 # seconds
for q in (xc, yc, a, b, XC, YC, R):
    q *= 0.42 # micrometers

area = np.pi*a*b
AREA = np.pi*R**2

xc -= np.mean(xc)
yc -= np.mean(yc)
XC -= np.mean(XC)
YC -= np.mean(YC)

area -= np.mean(area)
AREA -= np.mean(AREA)
    
    
fig, ax = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
ax = ax.ravel()

# Look at centers
ax[0].set_ylabel('Xc [$\mu m$]', color='red')
ax[0].plot(time, xc, color='red')
ax[0].plot(time, sfilter(xc), color='black')

ax[1].set_ylabel('Xc [$\mu m$]', color='blue')
ax[1].plot(time, XC, color='blue')
ax[1].plot(time, sfilter(XC), color='black')

ax[2].set_ylabel('Yc [$\mu m$]', color='red')
ax[2].plot(time, yc, color='orange')
ax[2].plot(time, sfilter(yc), color='black')

ax[3].set_ylabel('Yc [$\mu m$]', color='blue')
ax[3].plot(time, YC, color='cyan')
ax[3].plot(time, sfilter(YC), color='black')

# Look at radii
ax[4].set_ylabel('Axis [$\mu m$]', color='red')
ax[4].plot(time, a, color='red')
ax[4].plot(time, sfilter(a), color='black')
ax[4].plot(time, b, color='orange')
ax[4].plot(time, sfilter(b), color='black')

ax[5].set_ylabel('Axis [$\mu m$]', color='blue')
ax[5].plot(time, R, color='blue')
ax[5].plot(time, sfilter(R), color='black')

# Areas
ax[6].set_ylabel('Area [$\mu m^2$]', color='red')
ax[6].plot(time, area, color='red')
ax[6].plot(time, sfilter(area), color='black')
ax[6].set_xlabel('Time [s]')

ax[7].set_ylabel('Area [$\mu m^2$]', color='blue')
ax[7].plot(time, AREA, color='blue')
ax[7].plot(time, sfilter(AREA), color='black')
ax[7].set_xlabel('Time [s]')

plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.1)

plt.show()


