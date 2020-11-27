from array2gif import write_gif
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


gif_path = './data/artery_REM.gif'
imgs = imageio.mimread(gif_path)
# In the original movie there is one vessel that can be used to get
# radius changes in time
# Crop and ignore the last transparency channel
imgs = [img[120:220, 180:280, :3] for img in imgs]
reds = [img[:, :, 0] for img in imgs]
greens = [img[:, :, 1] for img in imgs]


for i, (img, red, green) in enumerate(zip(imgs, reds, greens)):
    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()

    ax[0].imshow(img)
    ax[0].set_xlabel('Full')
    
    ax[1].imshow(green)
    ax[1].set_xlabel('Green')

    red[green > red] = 0
    red = red*(green < 10)
    
    ax[2].imshow(red)
    ax[2].set_xlabel('Red')

    plt.title('Step {}'.format(i))
    fig.tight_layout()

    fig.savefig('./data/problem_{:03d}'.format(i))
