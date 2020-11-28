import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def sfilter(array):
    '''Low pass filter'''
    b, a = signal.butter(3, 0.33/2.)
    y = signal.filtfilt(b, a, array)
    return y


def min_mean_max(y, t, ax, colors):
    ax.plot(t, np.min(y)*np.ones_like(t), linestyle='dashed', color=colors[0])
    ax.text(t[200], np.min(y), '{:0.2f}'.format(np.min(y)),
            color=colors[0],
            ha="right", va="top",
            bbox=dict(boxstyle="square", alpha=0.7)
    )
    
    ax.plot(t, np.mean(y)*np.ones_like(t), linestyle='dashed', color=colors[1])
    ax.text(t[len(t)-200], np.mean(y), '{:0.2f}'.format(np.mean(y)),
            color=colors[1],
            ha="right", va="top",
            bbox=dict(boxstyle="square", alpha=0.7)
    )    
    
    ax.plot(t, np.max(y)*np.ones_like(t), linestyle='dashed', color=colors[2])
    ax.text(t[len(t)-1], np.max(y), '{:0.2f}'.format(np.max(y)),
            color=colors[2],
            ha="right", va="top",
            bbox=dict(boxstyle="square", alpha=0.7)
    )    


# Image size is
x0, y0 = 103//2, 110//2

# Blood
data = np.loadtxt('./blood/log.txt')

time, xc, yc, a, b, theta0, XC, YC, R = data.T
time *= 0.33 # seconds
for q in (xc, yc, a, b, XC, YC, R):
    q *= 0.42 # micrometers

area = np.pi*a*b
AREA = np.pi*R**2
# With respect to image center
XC -= x0
YC -= y0

# ----------------------

data = np.loadtxt('./tissue/log.txt')

timet, XCt, YCt, Rt = data.T
timet *= 0.33 # seconds
for q in (XC, YC, R):
    q *= 0.42 # micrometers

AREAt = np.pi*Rt**2
# With respect to image center
XCt -= x0
YCt -= y0

# ------------
    
fig, ax = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
ax = ax.ravel()

# Look at centers
ax[0].set_title('Tissue', color='limegreen')
ax[0].set_ylabel('Xc-X0 [$\mu m$]', color='limegreen')
ax[0].plot(time, XCt, color='limegreen')
min_mean_max(sfilter(XCt), time, ax[0], ['tan', 'orange', 'gold'])
ax[0].plot(time, sfilter(XCt), color='black')

ax[2].set_ylabel('Yc-Y0 [$\mu m$]', color='limegreen')
ax[2].plot(time, YCt, color='limegreen')
min_mean_max(sfilter(YCt), time, ax[2], ['tan', 'orange', 'gold'])
ax[2].plot(time, sfilter(YCt), color='black')

ax[4].set_ylabel('Radius [$\mu m$]', color='limegreen')
ax[4].plot(time, Rt, color='limegreen')
min_mean_max(sfilter(Rt), time, ax[4], ['tan', 'orange', 'gold'])
ax[4].plot(time, sfilter(Rt), color='black')

ax[6].set_ylabel('Area [$\mu m^2$]', color='limegreen')
ax[6].plot(time, AREAt, color='limegreen')
min_mean_max(sfilter(AREAt), time, ax[6], ['tan', 'orange', 'gold'])
ax[6].plot(time, sfilter(AREAt), color='black')
ax[6].set_xlabel('Time [s]')

# -----------
ax[1].set_title('Vessel', color='tomato')
ax[1].set_ylabel('Xc-X0 [$\mu m$]', color='tomato')
ax[1].plot(timet, XC, color='tomato')
min_mean_max(sfilter(XC), time, ax[1], ['tan', 'orange', 'gold'])
ax[1].plot(timet, sfilter(XC), color='black')

ax[3].set_ylabel('Yc-Y0 [$\mu m$]', color='tomato')
ax[3].plot(timet, YC, color='tomato')
min_mean_max(sfilter(YC), time, ax[3], ['tan', 'orange', 'gold'])
ax[3].plot(timet, sfilter(YC), color='black')

ax[5].set_ylabel('Radius [$\mu m$]', color='tomato')
ax[5].plot(timet, R, color='tomato')
min_mean_max(sfilter(R), time, ax[5], ['tan', 'orange', 'gold'])
ax[5].plot(timet, sfilter(R), color='black')

ax[7].set_ylabel('Area [$\mu m^2$]', color='tomato')
ax[7].plot(timet, AREA, color='tomato')
min_mean_max(sfilter(AREA), time, ax[7], ['tan', 'orange', 'gold'])
ax[7].plot(timet, sfilter(AREA), color='black')
ax[7].set_xlabel('Time [s]')

plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.1)

# Comparison unfiltered filtered

fig, ax = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
ax = ax.ravel()

# Look at centers
ax[0].set_title('Raw')
ax[0].set_ylabel('Xc-X0 [$\mu m$]', color='limegreen')
ax[0].plot(time, XCt, color='limegreen')
ax_ = ax[0].twinx()
ax_.set_ylabel('Xc-X0 [$\mu m$]', color='tomato')
ax_.plot(time, XC, color='tomato')

ax[2].set_ylabel('Yc-Y0 [$\mu m$]', color='limegreen')
ax[2].plot(time, YCt, color='limegreen')
ax_ = ax[2].twinx()
ax_.set_ylabel('Yc-Y0 [$\mu m$]', color='tomato')
ax_.plot(time, YC, color='tomato')

ax[4].set_ylabel('Radius [$\mu m$]', color='limegreen')
ax[4].plot(time, Rt, color='limegreen')
ax_ = ax[4].twinx()
ax_.set_ylabel('Radius [$\mu m$]', color='tomato')
ax_.plot(time, R, color='tomato')

ax[6].set_ylabel('Area [$\mu m^2$]', color='limegreen')
ax[6].plot(time, AREAt, color='limegreen')
ax_ = ax[6].twinx()
ax_.plot(time, AREA, color='tomato')
ax_.set_ylabel('Area [$\mu m^2$]', color='tomato')
ax_.set_xlabel('Time [s]')

# -----------

ax[1].set_title('Filtered')
ax[1].set_ylabel('Xc-X0 [$\mu m$]', color='limegreen')
ax[1].plot(time, sfilter(XCt), color='limegreen')
ax_ = ax[1].twinx()
ax_.set_ylabel('Xc-X0 [$\mu m$]', color='tomato')
ax_.plot(time, sfilter(XC), color='tomato')

ax[3].set_ylabel('Yc-Y0 [$\mu m$]', color='limegreen')
ax[3].plot(time, sfilter(YCt), color='limegreen')
ax_ = ax[3].twinx()
ax_.set_ylabel('Yc-Y0 [$\mu m$]', color='tomato')
ax_.plot(time, sfilter(YC), color='tomato')

ax[5].set_ylabel('Radius [$\mu m$]', color='limegreen')
ax[5].plot(time, sfilter(Rt), color='limegreen')
ax_ = ax[5].twinx()
ax_.set_ylabel('Radius [$\mu m$]', color='tomato')
ax_.plot(time, sfilter(R), color='tomato')

ax[7].set_ylabel('Area [$\mu m^2$]', color='limegreen')
ax[7].plot(time, sfilter(AREAt), color='limegreen')
ax_ = ax[7].twinx()
ax_.plot(time, sfilter(AREA), color='tomato')
ax_.set_ylabel('Area [$\mu m^2$]', color='tomato')
ax_.set_xlabel('Time [s]')


plt.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.1)


plt.show()


