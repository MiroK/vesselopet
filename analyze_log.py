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

    
def subset_mean(y, t, ax, pieces=((0, 210), (210, 310), (310, 370), (370, 400), (400, 1787)),
                colors=('blue', 'dodgerblue', 'deepskyblue', 'steelblue', 'darkblue')):
    for color, (t0, t1) in zip(colors, pieces):
        idx, = np.where(np.logical_and(t0 < t, t < t1))
        t_ = t[idx]
        y_ = np.mean(y[idx])
        ax.plot(t_, y_*np.ones_like(t_))
        ax.text(0.5*(t_[0]+t_[-1]), y_, '{:0.2f}'.format(y_),
                color=color,
                ha="right", va="top",
                bbox=dict(boxstyle="square", alpha=0.7, fc='gold'))
    

# Image size is
x0, y0 = 103//2, 110//2

# Blood
data0 = np.loadtxt('./combined/log.txt')
with open('./combined/log.txt') as log:
    keys = log.readline()[1:].strip().split(' ')
    data0 = {key: data0[:, i] for i, key in enumerate(keys)}
    
time, XC, YC, R = data0['time'], data0['blood_xc'], data0['blood_yc'], data0['blood_r']
time *= 0.33 # seconds
for q in (XC, YC, R):
    q *= 0.42 # micrometers

AREA = np.pi*R**2
# With respect to image center
XC -= x0
YC -= y0

# ----------------------

data1 = np.loadtxt('./combined/log.txt')  # This could be potentially different
with open('./combined/log.txt') as log:
    keys = log.readline()[1:].strip().split(' ')
    data1 = {key: data1[:, i] for i, key in enumerate(keys)}
    
timet, XCt, YCt, Rt = data1['time'], data1['tissue_xc'], data1['tissue_yc'], data1['tissue_r']
timet *= 0.33 # seconds
for q in (XCt, YCt, Rt):
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
fig.savefig('tissue_vessel_raw.pdf')

# Comparison unfiltered filtered

fig, ax = plt.subplots(6, 2, figsize=(16, 10), sharex=True)
ax = ax.ravel()

# Look at centers
ax[0].set_title('Raw')
ax[0].set_ylabel('Xc-X0 [$\mu m$]', color='limegreen', rotation=80)
ax[0].plot(time, XCt, color='limegreen')
ax_ = ax[0].twinx()
ax_.set_ylabel('Xc-X0 [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, XC, color='tomato')

ax[2].set_ylabel('Yc-Y0 [$\mu m$]', color='limegreen', rotation=80)
ax[2].plot(time, YCt, color='limegreen')
ax_ = ax[2].twinx()
ax_.set_ylabel('Yc-Y0 [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, YC, color='tomato')

ax[4].set_ylabel('Radius [$\mu m$]', color='limegreen', rotation=80)
ax[4].plot(time, Rt, color='limegreen')
#subset_mean(Rt, time, ax[4])
ax_ = ax[4].twinx()
ax_.set_ylabel('Radius [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, R, color='tomato')
#subset_mean(R, time, ax_)

ax[6].set_ylabel('$\Delta$ Radius [$\mu m$]', color='black', rotation=80)
ax[6].plot(time, Rt-R, color='black')
subset_mean(Rt-R, time, ax[6])

ax[8].set_ylabel('Area [$\mu m^2$]', color='limegreen', rotation=80)
ax[8].plot(time, AREAt, color='limegreen')
#subset_mean(AREAt, time, ax[8])
ax_ = ax[8].twinx()
ax_.plot(time, AREA, color='tomato')
ax_.set_ylabel('Area [$\mu m^2$]', color='tomato', rotation=80)
#subset_mean(AREA, timet, ax_)

ax[10].set_ylabel('$\Delta$ Area [$\mu m^2$]', color='black', rotation=80)
ax[10].plot(time, AREAt-AREA, color='black')
subset_mean(AREAt-AREA, time, ax[10])
ax[10].set_xlabel('Time [s]')

# -----------

ax[1].set_title('Filtered')
ax[1].set_ylabel('Xc-X0 [$\mu m$]', color='limegreen', rotation=80)
ax[1].plot(time, sfilter(XCt), color='limegreen')
ax_ = ax[1].twinx()
ax_.set_ylabel('Xc-X0 [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, sfilter(XC), color='tomato')

ax[3].set_ylabel('Yc-Y0 [$\mu m$]', color='limegreen', rotation=80)
ax[3].plot(time, sfilter(YCt), color='limegreen')
ax_ = ax[3].twinx()
ax_.set_ylabel('Yc-Y0 [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, sfilter(YC), color='tomato')

ax[5].set_ylabel('Radius [$\mu m$]', color='limegreen', rotation=80)
ax[5].plot(time, sfilter(Rt), color='limegreen')
ax_ = ax[5].twinx()
ax_.set_ylabel('Radius [$\mu m$]', color='tomato', rotation=80)
ax_.plot(time, sfilter(R), color='tomato')

ax[7].set_ylabel('$\Delta$ Radius [$\mu m$]', color='black', rotation=80)
ax[7].plot(time, sfilter(Rt)-sfilter(R), color='black')
subset_mean(sfilter(Rt)-sfilter(R), time, ax[7])

ax[9].set_ylabel('Area [$\mu m^2$]', color='limegreen', rotation=80)
ax[9].plot(time, sfilter(AREAt), color='limegreen')
ax_ = ax[9].twinx()
ax_.plot(time, sfilter(AREA), color='tomato')
ax_.set_ylabel('Area [$\mu m^2$]', color='tomato', rotation=80)

ax[11].set_ylabel('$\Delta$ Area [$\mu m^2$]', color='black', rotation=80)
ax[11].plot(time, sfilter(AREAt)-sfilter(AREA), color='black')
subset_mean(sfilter(AREAt)-sfilter(AREA), time, ax[11])
ax[11].set_xlabel('Time [s]')

plt.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.2)
fig.savefig('comparison_raw.pdf')

plt.show()


