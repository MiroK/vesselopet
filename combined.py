# Estimator img -> auxiliary image, shape
# shape parameters
from skimage.measure import CircleModel
from convex_hull import array_convex_hull
from edges import RadialFinder, VesselWidthFinder, TissueWidthFinder
from edges import reduce_last
import numpy as np


class VesselWidth(object):
    name = 'VesselWidth'
    def __init__(self, img, src, dst, prefix='', **kwargs):
        assert False  # FIXME
        # For logging columns
        self.params_names = ('xc', 'yc', 'r')
        if prefix:
            self.params_names = tuple('_'.join([prefix, p])
                                      for p in self.params_names)
        # Values to be updated at fit
        self.params = ()
    
        self.m = VesselWidthFinder(img, src, dst, **kwargs)

    def fit(self, image):
        (i0, j0, v0), (i1, j1, v1) = self.m.sample(image)

        B = np.zeros_like(image)
        B[i0, j0] = 1
        B[i1, j1] = 1

        self.params = (i0+i1)//2, (j0+j1)//2, 0.5*np.sqrt((i0-i1)**2 + (j0-j1)**2)

        return B, np.array([[i0, j0],
                            [i1, j1]])
    
    
class TissueWidth(object):
    name = 'TissueWidth'
    def __init__(self, img, src, dst, prefix='', **kwargs):
        # For logging columns
        self.params_names = ('xc', 'yc', 'r')
        if prefix:
            self.params_names = tuple('_'.join([prefix, p])
                                      for p in self.params_names)
        # Values to be updated at fit
        self.params = ()
    
        self.m = TissueWidthFinder(img, src, dst, **kwargs)

    def fit(self, image):
        (i0, j0, v0), (i1, j1, v1) = self.m.sample(image)

        B = np.zeros_like(image)
        B[i0, j0] = 1
        B[i1, j1] = 1

        self.params = (i0+i1)//2, (j0+j1)//2, 0.5*np.sqrt((i0-i1)**2 + (j0-j1)**2)

        return B, np.array([[i0, j0],
                            [i1, j1]])

    
class TissueRadialEstimator(object):
    name = 'TissueRadialEstimator'
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


class VesselRadialEstimator(TissueRadialEstimator):
    name = 'VesselRadialEstimator'
    def __init__(self, img, nthetas=200, prefix=''):
        super(VesselRadialEstimator, self).__init__(img, nthetas, prefix)
    
        self.m = RadialFinder(f=lambda x: x,
                              reduction=lambda x: reduce_last(x, 0.5*np.max(x)),
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
    name = 'CVXHullEstimator'
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
        # Need a better model here 
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
    from filters import normalize, split_jobs, time_smooth
    from mpi4py import MPI
    import tqdm, os, argparse
    import matplotlib
    import matplotlib.pyplot as plt
    import skimage.io as io
    
    matplotlib.use('AGG') 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-smoothed', type=int, default=0, help='Time smoothing')
    args = parser.parse_args()

    rpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch02_short_video-1.tif' 
    red_seq = io.imread(rpath)

    gpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch01_short_video-1.tif' 
    green_seq = io.imread(gpath)

    smoothed = bool(args.smoothed)

    green_nseq = normalize(green_seq)
    green_nseq_bk = 1*green_nseq

    red_nseq = normalize(red_seq)
    red_nseq_bk = 1*red_nseq
    
    if smoothed:
        green_nseq, (shift, _) = time_smooth(green_nseq, width=3)
        red_nseq, (shift, _) = time_smooth(red_nseq, width=3)
    else:
        shift = 0
    
    r = red_nseq[0]
    # Remove frame from the blood series
    lft, rght, tp, btm = [f(r, 25) for f in (left_edge, right_edge, top_edge, bottom_edge)]
    time = np.arange(len(red_nseq))
    for e in (lft, rght, tp, btm):
        red_nseq[np.ix_(time, e[0].flatten(), e[1].flatten())] = 0

    comm = MPI.COMM_WORLD

    blue = np.zeros_like(r)        

    # red_estimator = CVXHullEstimator(prefix='blood')
    # green_estimator = TissueRadialEstimator(img=blue, prefix='tissue')

    nrows, ncols = blue.shape
    # Green sample similar to laura
    src, dst = (0+5, 0), (nrows-1, ncols-1+5)
    green_estimator = TissueWidth(blue, src, dst, prefix='tissue')    
    red_estimator = VesselRadialEstimator(img=blue, prefix='blood')


    result_dir = '{}_{}_smooth{}'.format(green_estimator.name, red_estimator.name, smoothed)
    comm.rank == 0 and not os.path.isdir(result_dir) and os.mkdir(result_dir)

    time_idx = np.arange(len(time))
    my_times = split_jobs(comm, time_idx)

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
                               green_nseq_bk[i],
                               blue], axis=2))
        # Center title
        ax[1].set_title('Time {:0.2f} s'.format(i*0.33))

        # Tissue fit
        gimage = green_nseq[i]
        aux_green, green_mask = green_estimator.fit(gimage)
        ax[1].imshow(aux_green)
        # Add it's data
        ax[1].plot(green_mask[:, 1], green_mask[:, 0], color='gold')
        ax[2].plot(green_mask[:, 1], green_mask[:, 0], color='gold')
        
        # Blood fit
        rimage = red_nseq[i]

        aux_red, red_mask = red_estimator.fit(rimage)
        ax[0].imshow(aux_red)
        # Add it's data
        ax[0].plot(red_mask[:, 1], red_mask[:, 0], color='blue', linewidth=8)
        ax[2].plot(red_mask[:, 1], red_mask[:, 0], color='blue', linewidth=8)


        for axi in ax: axi.axis('off')
        # Next round
        fig.savefig('{}/img_{:04d}.png'.format(result_dir, i+shift))
        for axi in ax: axi.cla()

        row = (i+shift, ) + red_estimator.params + green_estimator.params
        with open(log_file, 'a') as log:
            log.write(fmt % row)
        results.append(row)

    results = comm.allgather(results)
    results = np.row_stack([np.array(r) for r in results])

    comm.rank == 0 and np.savetxt('{}/log.txt'.format(result_dir), results, header=names)

    # FIXME: green /
    #        blood ala Laura estimator X
