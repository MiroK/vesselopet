from skimage.measure import profile_line
import numpy as np
import itertools 


class RadialSampler(object):
    '''Sample image on lines going from (x0, y0) to boundaries'''
    def __init__(self, img, x0=None, y0=None):
        nrows, ncols = img.shape

        if x0 is None or y0 is None:
            x0, y0 = nrows//2, ncols//2
        assert 0 <= x0 < nrows and 0 <= y0 <= ncols

        self.cols = np.tile(np.arange(ncols), (nrows, 1))
        self.cols.dtype = int
        self.rows = np.tile(np.arange(nrows), (ncols, 1)).T
        self.rows.dtype = int
        self.src = (x0, y0)

    def sample(self, img):
        '''Image values on rays'''
        for dst in self.boundary_points:
            line_values = profile_line(img, self.src, dst, mode='constant')
            yield line_values

    @property
    def rays(self):
        '''Indices of points on rays'''
        yield from zip(self.sample(self.rows), self.sample(self.cols))

    @property
    def boundary_points(self):
        '''Boundary indices of the image from 00 clockwise'''
        rows, cols = self.rows, self.cols
        yield from itertools.chain(zip(rows[0], cols[0]),
                                   zip(rows[1:-1, -1], cols[1:-1, -1]),
                                   zip(reversed(rows[-1]), reversed(cols[-1])),
                                   zip(reversed(rows[1:-1, 0]), reversed(cols[1:-1, 0])))


class RadialFinder(RadialSampler):
    '''Sampling rays results in indices reduce(f(ray))'''
    def __init__(self, f, reduction, img, x0=None, y0=None):
        super(RadialFinder, self).__init__(img, x0, y0)
        self.f = f
        self.reduce = reduction

    def find(self, img):
        '''Where and what are f on ray satisfying predicate'''
        for indices, ray in zip(self.rays, super(RadialFinder, self).sample(img)):
            ray_values = self.f(ray)
            true_idx = self.reduce(ray_values)
            if true_idx:
                yield (np.array(indices[0][true_idx], dtype=int),
                       np.array(indices[1][true_idx], dtype=int),
                       ray_values[true_idx])

    def collect(self, img):
        '''Just indices'''
        indices = self.find(img)

        c0, c1, _ = next(indices)
        i0, i1 = [c0], [c1]
        for c0, c1, _ in indices:
            i0.append(c0)
            i1.append(c1)
        return np.hstack(i0), np.hstack(i1)


def reduce_first(x, value):
    '''First index where x > value'''
    idx, = np.where(x > value)
    if len(idx) == 0:
        return ()
    return (idx[0], )


def reduce_last(x, value):
    '''Last index where x > value'''
    idx, = np.where(x > value)
    if len(idx) == 0:
        return ()
    return (idx[-1], )


def reduce_mean(x, value):
    '''Mean of index where x > value'''
    idx, = np.where(x > value)
    if len(idx) == 0:
        return ()
    return (int(np.mean(idx)), )


def reduce_middle(x, value):
    '''Mean of index where x > value'''
    idx, = np.where(x > value)
    if len(idx) == 0:
        return ()
    return (int(0.5*(idx[0]+idx[-1])), )


def reduce_wmean(x, value):
    '''Mean of index where x > value. Weighed by distance'''
    idx, = np.where(x > value)
    if len(idx) == 0:
        return ()
    weights = 1./idx
    return (int(sum(idx*weights)/sum(weights)), )

    
class MaxFinder(RadialFinder):
    def __init__(self, img, x0=None, y0=None):
        super(MaxFinder, self).__init__(lambda x: x, np.argmax, img, x0, y0)
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    from skimage.measure import EllipseModel, CircleModel
        
    ellipse, circle = EllipseModel(), CircleModel()

    from filters import normalize
    import matplotlib
    #matplotlib.use('AGG') 
    import matplotlib.pyplot as plt

    import skimage.io as io

    rpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch02_short_video-1.tif' 
    red_seq = io.imread(rpath)

    red_nseq = normalize(red_seq)

    gpath = '/home/mirok/Downloads/MIRO_TSeries-01302019-0918-028_cycle_001_ch01_short_video-1.tif' 
    green_seq = io.imread(gpath)

    green_nseq_bk = normalize(green_seq)

    m = RadialFinder(f=lambda x: x,
                     reduction=lambda x: np.argmax(x),#reduce_middle(x, 0.6*max0),
                     img=green_nseq_bk[0])

    blue = np.zeros_like(green_nseq_bk[0])
    
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()

    import tqdm
    import os

    not os.path.isdir('tissue') and os.mkdir('tissue')

    with open('tissue/log.txt', 'w') as out:
        out.write('# t XC YC R\n')
    
    for idx in tqdm.tqdm(range(len(green_nseq_bk))):
        img = green_nseq_bk[idx]
        max0 = np.max(img)    

        ax[0].imshow(np.stack([red_nseq[idx], img, blue], axis=2))

        results = np.zeros_like(img)
        x_, y_ = m.collect(img)
        ii, = np.where(np.logical_and(np.logical_and(x_ > 10, x_ < 80),
                                      np.logical_and(y_ > 10, y_ < 80)))
        x_ = x_[ii]
        y_ = y_[ii]
        results[x_, y_] = 1

        ax[1].imshow(results)

        circle.estimate(np.c_[x_, y_])

        theta = np.linspace(0, 2*np.pi, 200)
        XC, YC, R = circle.params
        x = XC + R*np.sin(theta)
        y = YC + R*np.cos(theta)

        with open('tissue/log.txt', 'a') as out:
            out.write('{} {} {} {}\n'.format(idx, XC, YC, R))

        for axi in ax:
            axi.plot(y, x, color='orange')

   
        fig.savefig('tissue/img_{:04d}.png'.format(idx))
        for axi in ax:
            axi.cla()

    plt.show()
