from edges import RadialFinder, reduce_last

from skimage.measure import CircleModel
from skimage.io import ImageCollection
import numpy as np
import imageio


def read_gif(path):
    '''Break up gif'''
    return list(iter(imageio.get_reader(path).iter_data()))


class RadiusEstimator(RadialFinder):
    '''Estimate vessel radius by fitting a circle'''
    def __init__(self, img, threshold=0.2):
        # The main idea of the algo is based on assumed radiual "symmetry"
        # of the result. So when we fit the circle we don't want to work with
        # all the pixels by just the "rim" ones for each angle
        super(RadiusEstimator, self).__init__(
            f=lambda x: x,
            # For each ray, how do you find a rim?
            reduction=lambda x: reduce_last(x, threshold*np.max(x)),
            img=img)
        # And we will be fitting a circle
        self.model = CircleModel()
        
    def __call__(self, img):
        '''Estimate for the given image'''
        x_, y_ = self.collect(img)

        ii, = np.where(np.logical_and(np.logical_and(x_ > 10, x_ < 80),
                                      np.logical_and(y_ > 10, y_ < 80)))

        x_, y_ = x_[ii], y_[ii]
        self.model.estimate(np.c_[x_, y_])
        # Give back the circle parameters
        return self.model.params
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Each frame has four channels
    img_sequence = read_gif('./data/red.gif')
    # We are after the red on
    red_sequence = [frame[:, :, 0] for frame in img_sequence] 

    idx = 24
    estimate = RadiusEstimator(img=red_sequence[idx])

    x0, y0, r0 = estimate(red_sequence[idx])

    thetas = np.linspace(0, 2*np.pi, 200)
    x = x0 + r0*np.sin(thetas)
    y = y0 + r0*np.cos(thetas)

    # Just for visual inspection
    fig, ax = plt.subplots()
    ax.imshow(red_sequence[idx])
    ax.plot(y, x, color='magenta')
    plt.show()

    # NOTE: Let's say that if this works then running rhough the sequence
    # we would generate a dataset for image-segmentation neural network
