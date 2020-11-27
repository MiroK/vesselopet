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
red = [img[:, :, 0] for img in imgs]
green = [img[:, :, 1] for img in imgs]

null = np.zeros_like(green[0])
# It's fortunate that we are in the RGB encoding
# Interstitium is pretty much green
write_gif([np.stack([null, g, null], axis=2) for g in green], './data/green.gif')  # NOTE: these reorder!

# The red should be flow but the channel actually has nonzero
# values also in the "green". So we don't just apply the red
# mask to the entire image but only to not green part. Not green is
# defined by tolerance
tol = 10

red_filtered = []
for g, r in zip(green, red):
    # Get only pixel where green intensity is low enough
    mask = g < tol
    red_filtered.append(r*mask)

write_gif([np.stack([r, null, null], axis=2) for r in red_filtered], './data/red.gif')  # NOTE: these reorder!


# # Vessel is the volume bounded by red
from convex_hull import array_convex_hull
from skimage.morphology import convex_hull_image


xx = array_convex_hull(red_filtered[0], 4)





