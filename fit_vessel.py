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
    from skimage.draw import disk
    import os, tqdm
    
    def make_segmentation_data(sequence, indices=None, show=False, save_dir='./data/segmentation'):
        '''Create input/output arrays for segmentation'''
        if indices is None:
            indices = range(len(sequence))

        if isinstance(indices, int):
            indices = (indices, )

        # Circle parametrization for plotting
        thetas = np.linspace(0, 2*np.pi, 200)
        # Setup esimator
        img = sequence[0]
        shape = img.shape
        
        estimate = RadiusEstimator(img=img)
        null = np.zeros_like(img)

        not os.path.exists(save_dir) and os.makedirs(save_dir)        
        
        nn_inputs, nn_ouputs, circles = [], [], []
        for idx in tqdm.tqdm(indices):
            nn_input = red_sequence[idx]
            # Get the center
            x0, y0, r0 = estimate(nn_input)
            # Mask
            rr, cc = disk((x0, y0), r0, shape=shape)
            null[rr, cc] = 1

            # Collect
            nn_inputs.append(nn_input)
            nn_ouputs.append(1*null)
            circles.append((x0, y0, r0))

            if show:
                x, y = x0 + r0*np.sin(thetas), y0 + r0*np.cos(thetas)
                
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(nn_input)
                ax[0].plot(y, x, color='magenta')

                ax[1].imshow(null)

                path = os.path.join(save_dir, 'images')
                not os.path.exists(path) and os.mkdir(path)
                
                path = os.path.join(path, 'img{:04d}.png'.format(idx))
                plt.savefig(path)
                plt.close()

            # Get ready for next round
            null[:] *= 0
        # Done and save

        nn_inputs, nn_outputs, circles = map(np.array, (nn_inputs, nn_ouputs, circles))

        np.save(os.path.join(save_dir, 'nn_inputs.npy'), nn_inputs)
        np.save(os.path.join(save_dir, 'nn_outputs.npy'), nn_outputs)
        np.savetxt(os.path.join(save_dir, 'circles.txt'), circles, header='Center-x Center-y Radius')        

        print(f'Inputs {nn_inputs.shape} with {nn_inputs.dtype}')
        print(f'Outputs {nn_outputs.shape} with {nn_outputs.dtype}')        
        
    # Each frame has four channels
    img_sequence = read_gif('./data/red.gif')
    # We are after the red on
    red_sequence = [frame[:, :, 0] for frame in img_sequence] 

    # Just for illustration
    # make_segmentation_data(red_sequence, indices=(32, 43), show=True, save_dir='./data/segmentation')

    # All is `indices=None`
    make_segmentation_data(red_sequence, indices=range(10), show=True, save_dir='./data/segmentation')    
