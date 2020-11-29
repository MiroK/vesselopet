from skimage.measure import profile_line
from scipy import signal
import numpy as np
import itertools 

import matplotlib.pyplot as plt


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


class SegmentSampler(object):
    '''Sample image on lines going from (x0, y0) to boundaries'''
    def __init__(self, img, src, dst):
        nrows, ncols = img.shape

        cols = np.tile(np.arange(ncols), (nrows, 1))
        rows = np.tile(np.arange(nrows), (ncols, 1)).T

        i_values = np.array(profile_line(rows, src, dst, mode='constant'), dtype=int)
        j_values = np.array(profile_line(cols, src, dst, mode='constant'), dtype=int)

        self.i_values, self.j_values = i_values, j_values

        self.src = src
        self.dst = dst

    def sample(self, img):
        '''Image values on rays'''
        line_values = profile_line(img, self.src, self.dst, mode='constant')
        return (self.i_values, self.j_values, line_values)


class VesselWidthFinder(SegmentSampler):
    '''
    Estimate __|---|__ region (blood)

    The idea here is to sample on line and then consider separately halfs 
    as determined by line parametrization ( t \in(0, 1) ). On the t > 0.5 
    we want the most distant "peak". For t < 0.5 it is the closest one.
    '''
    def __init__(self, img, src, dst, nlpeaks=6, ltol=0.6, nrpeaks=6, rtol=0.6):
        self.nlpeaks = nlpeaks
        self.ltol = ltol
        self.nrpeaks = nrpeaks
        self.rtol = rtol

        super(VesselWidthFinder, self).__init__(img, src, dst)
        
    def sample(self, img):
        i, j, line_values = super().sample(img)
        c = len(line_values)//2

        plt.figure()
        plt.plot(np.diff(line_values))
        plt.plot(line_values)
        plt.plot(np.cumsum(line_values))
        plt.show()
        
        size = max(line_values)
        # The idea is to find the most distant "large" peak
        vals = line_values[:c]
        peaks, _ = signal.find_peaks(vals)

        i0 = np.argmax(vals[peaks])
        max_ = peaks[i0]
        first = (i[:c][max_], j[:c][max_], vals[max_])

        vals = line_values[c:]
        peaks, _ = signal.find_peaks(vals)
        i0 = np.argmax(vals[peaks[:-1]]-vals[peaks[1:]])
        max_ = peaks[i0]
        second = (i[c:][max_], j[c:][max_], vals[max_])

        return first, second


class TissueWidthFinder(SegmentSampler):
    '''
    Estimate _/\--/\-- region (tissue)

    The idea here is to sample on line and then consider separately halfs 
    as determined by line parametrization ( t \in(0, 1) ). 
    we want the most distant "peak". For t < 0.5 it is the closest one.
    '''
    def __init__(self, img, src, dst):
        super(TissueWidthFinder, self).__init__(img, src, dst)
        
    def sample(self, img):
        i, j, line_values = super().sample(img)
        c = len(line_values)//2

        # The idea is to find the most distant "large" peak
        vals = line_values[:c]
        max_ = np.argmax(vals)
        first = (i[:c][max_], j[:c][max_], vals[max_])

        vals = line_values[c:]
        max_ = np.argmax(vals)
        second = (i[c:][max_], j[c:][max_], vals[max_])
        
        return first, second

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

    from filters import time_smooth
    
    bar, _ = time_smooth(green_nseq_bk, width=3)
    foo, _ = time_smooth(red_nseq, width=3)    
    # bar, xx = time_smooth(green_nseq, width=5, arrow=np.max)

    idx = 1
    nrows, ncols =  foo[0].shape
    ss = SegmentSampler(foo[idx], (nrows-1, 0), (0, ncols-1))
    s = VesselWidthFinder(foo[idx], (nrows-1, 0), (0, ncols-1)) 
    # s = TissueWidthFinder(foo[idx], (0+5, 0), (nrows-1, ncols-1+5))
    (i0, j0, v0), (i1, j1, v1) = s.sample(foo[idx])

    from scipy import signal
    
    plt.figure()
    xx = ss.sample(foo[idx])[2]

    peakind, _ = signal.find_peaks(xx)
    
    plt.plot(xx)
    print(np.argmin(np.abs(xx-v0)), np.argmin(np.abs(xx-v1)))
    plt.plot(peakind, xx[peakind], marker='x', linestyle='none')
    plt.plot(np.argmin(np.abs(xx-v0)), v0, marker='o')
    plt.plot(np.argmin(np.abs(xx-v1)), v1, marker='o')        
    
    #img[int(i0), int(j0)] = 1.
    #img[int(i1), int(j1)] = 1.
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(foo[idx])
    ax[0].plot([j0, j1], [i0, i1], color='magenta')

    ax[1].imshow(green_nseq_bk[idx])
    ax[1].plot([j0, j1], [i0, i1], color='magenta')
    
    # plt.show()
    
    #exit()
    
    m = RadialFinder(f=lambda x: x,
                     reduction=lambda x: reduce_last(x, 0.5*np.max(x)),# : np.argmax(x),#reduce_middle(x, 0.6*max0),
                     img=green_nseq_bk[0])

    blue = np.zeros_like(green_nseq_bk[0])
    
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()

    red_ = foo[idx]
    results = np.zeros_like(red_)
    x_, y_ = m.collect(red_)
    ii, = np.where(np.logical_and(np.logical_and(x_ > 10, x_ < 80),
                                  np.logical_and(y_ > 10, y_ < 80)))
    x_ = x_[ii]
    y_ = y_[ii]
    results[x_, y_] = 1

    circle = CircleModel()
    circle.estimate(np.c_[x_, y_])

    thetas = np.linspace(0, 2*np.pi, 200)
    XC, YC, R = circle.params
    x = XC + R*np.sin(thetas)
    y = YC + R*np.cos(thetas)

    ax[0].imshow(red_)
    ax[1].imshow(results)

    ax[0].plot(y, x, color='magenta')
    ax[1].plot(y, x, color='magenta')

    plt.show()
    exit()
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

# Consider running analysis on img integrated(mean, max) in time

# Radial sampling for red - after -1, most distant min?

# Red filter ala Laura            []
# Green filter ala Laura
