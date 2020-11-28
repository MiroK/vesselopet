# Estimator img -> auxiliary image, shape
# shape parameters
from skimage.measure import CircleModel
from convex_hull import array_convex_hull
from edges import RadialFinder
import numpy as np


class RadialEstimator(object):
    def __init__(self, img, nthetas=200, prefix=''):
        self.thetas = np.linspace(0, 2*np.pi, nthetas)
        # For logging columns
        self.params_names = ('xc', 'yc', 'r')
        if prefix:
            self.params_names = tuple('_'.join([prefix, p])
                                      for p in self.params_names)
        # Values to be updated at fit
        self.params = ()
    
        self.m = RadialFinder(f=lambda x: x,
                              reduction=lambda x: np.argmax(x),#reduce_middle(x, 0.6*max0),
                              img=img)

    def fit(self, image):
        B = np.zeros_like(image)
        # Radial sample for maximum
        x_, y_ = self.m.collect(image)
        # We are only after points that are not too far
        ii, = np.where(np.logical_and(np.logical_and(x_ > 10, x_ < 80),
                                      np.logical_and(y_ > 10, y_ < 80)))
        x_ = x_[ii]
        y_ = y_[ii]
        B[x_, y_] = 1

        circle = CircleModel()
        circle.estimate(np.c_[x_, y_])

        XC, YC, R = circle.params
        x = XC + R*np.sin(self.thetas)
        y = YC + R*np.cos(self.thetas)

        self.params = (XC, YC, R)

        return B, np.c_[x, y]


class CVXHullEstimator(object):
    '''Computes convex hull of mask and fits circle'''
    def __init__(self, nthetas=200, prefix=''):
        self.thetas = np.linspace(0, 2*np.pi, nthetas)
        # For logging columns
        self.params_names = ('xc', 'yc', 'r')
        if prefix:
            self.params_names = tuple('_'.join([prefix, p])
                                      for p in self.params_names)
        # Values to be updated at fit
        self.params = ()
    
    def fit(self, image):
        B = image > 0.5*np.max(image)#+3*np.std(image)
        
        circle = CircleModel()

        hull = array_convex_hull(B)    
        x_, y_ = hull.T
        x_ = np.r_[x_, x_[0]]
        y_ = np.r_[y_, y_[0]]
        circle.estimate(np.c_[x_, y_])
        
        # Plot ellipse
        XC, YC, R = circle.params
        x = XC + R*np.sin(self.thetas)
        y = YC + R*np.cos(self.thetas)

        self.params = (XC, YC, R)

        return B, np.c_[x, y]

# --------------------------------------------------------------------

if __name__ == '__main__':
    from filters import left_edge, right_edge, top_edge, bottom_edge
    from filters import normalize, split_jobs
    from mpi4py import MPI
    import tqdm
    import os
    
    import matplotlib
    matplotlib.use('AGG') 
    import matplotlib.pyplot as plt

    import skimage.io as io

    rpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch02_short_video-1.tif' 
    red_seq = io.imread(rpath)

    gpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch01_short_video-1.tif' 
    green_seq = io.imread(gpath)

    green_nseq = normalize(green_seq)
    
    red_nseq_bk = normalize(red_seq)
    red_nseq = 1*red_nseq_bk

    r = red_nseq[0]
    # Remove frame from the blood series
    lft, rght, tp, btm = [f(r, 25) for f in (left_edge, right_edge, top_edge, bottom_edge)]
    time = np.arange(len(red_nseq))
    for e in (lft, rght, tp, btm):
        red_nseq[np.ix_(time, e[0].flatten(), e[1].flatten())] = 0

    comm = MPI.COMM_WORLD

    result_dir = 'combined'
    comm.rank == 0 and not os.path.isdir(result_dir) and os.mkdir(result_dir)

    time_idx = np.arange(len(time))
    my_times = split_jobs(comm, time_idx)

    blue = np.zeros_like(r)        

    red_estimator = CVXHullEstimator(prefix='blood')
    green_estimator = RadialEstimator(img=blue, prefix='tissue')

    log_file = '{}/log_{}_{}.txt'.format(result_dir, comm.rank, comm.size)
    names = ' '.join(('time', ) + red_estimator.params_names + green_estimator.params_names)
    with open(log_file, 'w') as log:
        log.write('# {} \n'.format(names))
    fmt = '\t'.join(('%g', )*len(names.split(' '))) + '\n'

    results = []
    # I want a plot with auxiliary image for finding blood circle
    #                    auxiliary image for finding tissue circle
    #                    original data with the two circles
    fig, ax = plt.subplots(1, 3, figsize=(20, 14))
    for i in tqdm.tqdm(my_times):
        # Original image
        ax[2].imshow(np.stack([red_nseq_bk[i],
                               green_nseq[i],
                               blue], axis=2))
        # Center title
        ax[1].set_title('Time {:0.2f} s'.format(i*0.33))

        # Blood fit
        rimage = red_nseq[i]

        aux_red, red_mask = red_estimator.fit(rimage)
        ax[0].imshow(aux_red)
        # Add it's data
        ax[0].plot(red_mask[:, 1], red_mask[:, 0], color='blue')
        ax[2].plot(red_mask[:, 1], red_mask[:, 0], color='blue')

        # Tissue fit
        gimage = green_nseq[i]
        aux_green, green_mask = green_estimator.fit(gimage)
        ax[1].imshow(aux_green)
        # Add it's data
        ax[1].plot(green_mask[:, 1], green_mask[:, 0], color='gold')
        ax[2].plot(green_mask[:, 1], green_mask[:, 0], color='gold')

        for axi in ax: axi.axis('off')
        # Next round
        fig.savefig('{}/img_{:04d}.png'.format(result_dir, i))
        for axi in ax: axi.cla()

        row = (i, ) + red_estimator.params + green_estimator.params
        with open(log_file, 'a') as log:
            log.write(fmt % row)
        results.append(row)

    results = comm.allgather(results)
    results = np.row_stack([np.array(r) for r in results])

    comm.rank == 0 and np.savetxt('{}/log.txt'.format(result_dir), results, header=names)
