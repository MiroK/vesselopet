from array2gif import write_gif
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


def center_of_mass(img, tol=0, weighted=True):
    idx0, idx1 = np.where(img > tol)

    weights = img[idx0, idx1]
    if not weighted:
        weights[:] = 1
    points = np.c_[idx0, idx1]

    mass = np.sum(weights)
    centers = np.sum(points*weights[:, np.newaxis], axis=0)

    return centers/mass


gif = [np.array(g[:, :, 1]) for g in imageio.mimread('data/green.gif')]

reds = [np.array(g[:, :, 0]) for g in imageio.mimread('data/red.gif')]

from skimage.filters import gaussian

from skimage import feature
from sklearn.cluster import KMeans

idx = 12
for idx in np.arange(len(reds)):
    img = reds[idx]

    d = 10
    
    shape = img.shape
    img[:, tuple(range(d))] = 0
    img[:, tuple(range(shape[0]-d, shape[0]))] = 0
    img[tuple(range(d)), :] = 0
    img[tuple(range(shape[0]-d, shape[0])), :] = 0
    
    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()

    ax[0].imshow(img)

    X = np.array(np.where(img > 0)).T
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X)

    foo = np.zeros_like(img)
    foo[X[:, 0], X[:, 1]] = (kmeans.labels_ + 1)*100
    ax[1].imshow(foo)

    for c, l in zip(kmeans.cluster_centers_, kmeans.labels_):
        ax[1].plot(c[1], c[0], marker='x', color='magenta')

    kmeans = KMeans(n_clusters=1, random_state=0).fit(kmeans.cluster_centers_)
    bar = np.zeros_like(img)
    bar[X[:, 0], X[:, 1]] = (kmeans.predict(X) + 1)*100

    ax[2].imshow(bar)

    fig.savefig('kmeans_{:03d}.png'.format(idx))
#plt.colorbar()
