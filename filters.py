# Each filter taken an image and return insices where the filter should 
# be applied
import numpy as np
import itertools
import functools
import operator
import os

def left_edge(img, width):
    '''First width pixels from the left'''
    nrows, ncols = img.shape
    return np.ix_(np.arange(0, nrows), np.arange(0, width))


def right_edge(img, width):
    '''Last width pixels from the right'''
    nrows, ncols = img.shape
    return np.ix_(np.arange(0, nrows), np.arange(ncols-width, ncols))


def top_edge(img, width):
    '''First width pixels from the top'''
    nrows, ncols = img.shape
    return np.ix_(np.arange(0, width), np.arange(0, ncols))


def bottom_edge(img, width):
    '''Last width pixels from the bottom'''
    nrows, ncols = img.shape
    return np.ix_(np.arange(nrows-width, nrows), np.arange(0, ncols))


def border(img, width):
    '''Set border pixel of image to value'''
    indices = set(itertools.chain(itertools.product(idx[0].flatten(), idx[1].flatten())
                                  for idx in (left_edge(img, width),
                                              right_edge(img, width),
                                              top_edge(img, width),
                                              bottom_edge(img, width))))
    return list(indices)


def idx_expand(np_ix):
    '''Force cartesian product'''
    yield from itertools.product(np_ix[0].flatten(), np_ix[1].flatten())

    
def idx_union(indices):
    '''Combine indices'''
    yield from set(itertools.chain(*map(idx_expand, indices)))

    
def set_border(img, width, value=None):
    '''Set border pixel of image to value'''
    if value is None:
        value = next(np.zeros_like(img).flat)
    for f in (left_edge, right_edge, top_edge, bottom_edge):
        img[f(img, width)] = value
    return img


def tmean_threshold(seq, threshold, reduce=np.max):
    '''
    For (tdim, nx, ny) image whose pixels temporal mean is greater 
    then threshold * red(seq).
    '''
    mean = np.mean(seq, axis=0)
    max_value = reduce(seq)
    return np.where(mean > threshold*max_value)


def normalize(seq, reduce=np.max):
    '''Normalize such that reduce(space_t) is 1'''
    return seq/reduce(seq, axis=(1, 2))[:, np.newaxis, np.newaxis]
# Isolated pixels in space (by convolution? skimage.threshold.)
# Normalize seqeunce
#

def split_jobs(comm, jobs):
    '''Divide jobs(indices) over communicator'''
    njobs = len(jobs)
    nprocs = comm.size
    rank = comm.rank
        
    assert 0 <= rank < nprocs

    size = njobs//nprocs

    first = rank*size
    last = njobs if rank == nprocs-1 else (rank+1)*size

    my_jobs = [jobs[i] for i in range(first, last)]
    assert my_jobs

    return my_jobs

# --------------------------------------------------------------------

if __name__ == '__main__':
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
    # Too wide?
    lft, rght, tp, btm = [f(r, 25) for f in (left_edge, right_edge, top_edge, bottom_edge)]
    time = np.arange(len(red_nseq))
    for e in (lft, rght, tp, btm):
        red_nseq[np.ix_(time, e[0].flatten(), e[1].flatten())] = 0

    from skimage.filters import threshold_local
    from skimage.exposure import histogram
    from skimage.measure import EllipseModel, CircleModel
    from convex_hull import array_convex_hull
    from mpi4py import MPI
    import tqdm

    comm = MPI.COMM_WORLD
    
    comm.rank == 0 and not os.path.isdir('blood') and os.mkdir('blood')

    log_file = 'blood/log_{}_{}.txt'.format(comm.rank, comm.size)
    with open(log_file, 'w') as log:
        log.write('# i xc yc a b theta0 XC YC R\n')
    fmt = '\t'.join(('%g', )*9) + '\n'

    time_idx = np.arange(len(time))
    my_times = split_jobs(comm, time_idx)

    results = []
    fig, ax = plt.subplots(1, 3, figsize=(20, 14))        
    for i in tqdm.tqdm(my_times):
        ax[1].set_title('Time {} s'.format(i*0.33))
        image = red_nseq[i]

        #hist, hist_centers = histogram(image, nbins=10)
        #plt.figure()
        #plt.plot(hist_centers, hist)
    
        B = image > 0.5*np.max(image)#+3*np.std(image)
        
        ellipse, circle = EllipseModel(), CircleModel()

        hull = array_convex_hull(B)    
        x_, y_ = hull.T
        x_ = np.r_[x_, x_[0]]
        y_ = np.r_[y_, y_[0]]
        ellipse.estimate(np.c_[x_, y_])
        circle.estimate(np.c_[x_, y_])

        # Plot original
        ax[0].imshow(red_nseq_bk[i])
        # Auxiliary
        ax[1].imshow(B)

        blue = np.zeros_like(image)
        ax[2].imshow(np.stack([red_nseq_bk[i],
                               green_nseq[i],
                               blue], axis=2))
        
        # Plot ellipse
        theta = np.linspace(0, 2*np.pi, 200)

        xc, yc, a, b, theta0 = ellipse.params

        x = xc + a*np.sin(theta + theta0)
        y = yc + b*np.cos(theta + theta0)

        for axi in ax:
            axi.plot(y, x, color='blue')

        XC, YC, R = circle.params
        x = XC + R*np.sin(theta + theta0)
        y = YC + R*np.cos(theta + theta0)

        row = (i, xc, yc, a, b, theta0, XC, YC, R)
        with open(log_file, 'a') as log:
            log.write(fmt % row)
        results.append(row)
            
        for axi in ax:
            axi.plot(y, x, color='magenta')

        for axi in ax:
            axi.plot(y_, x_, color='cyan')

        for axi in ax:
            axi.axis('off')
            
        fig.savefig('blood/img_{:04d}.png'.format(i))
        for axi in ax:
            axi.cla()

    # Time info to video
    # Left is ellipse, right is circle
    # Center position
    # Axes length, radius
    # Area

    results = comm.allgather(results)
    results = np.row_stack([np.array(r) for r in results])

    if comm.rank == 0:
        np.savetxt('blood/log.txt', results, header='i xc yc a b theta0 XC YC R')
    #plt.show()
    # Threshold
    exit()

    green_nseq = normalize(green_seq)
    blue_nseq = np.zeros_like(green_nseq)

    from skimage import img_as_ubyte
    from skimage.feature import canny
    from skimage.transform import hough_ellipse
    from skimage.draw import ellipse_perimeter

    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)    

    # from skimage import measure
    # from skimage.segmentation import active_contour
    
    # idx = 100
    # g = green_nseq[idx]
    
    # contours = measure.find_contours(g, 1.0)
    # # Display the image and plot all contours found
    # fig, ax = plt.subplots()
    # ax.imshow(g)

    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


    # s = np.linspace(0, 2*np.pi, 400)
    # r = g.shape[0]//2 + 35*np.sin(s)
    # c = g.shape[1]//2 + 35*np.cos(s)
    # init = np.array([r, c]).T

    # snake = active_contour(g, init,
    #                        alpha=0.1, beta=1.0, w_line=0, w_edge=1, gamma=0.1)
    #                        #init, alpha=0.015, beta=10, gamma=0.001)


    # ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)

    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    # # v = np.stack([red_nseq, green_nseq, blue_nseq], axis=3)

    # # import os
    # # not os.path.isdir('full') and os.mkdir('full')

    # # fig, ax = plt.subplots()
    # # ax.axis('off')
    
    # # for i in range(len(v)):
    # #    ax.imshow(v[i])
    # #    fig.savefig('full/img_{:04d}.png'.format(i))
    # #    plt.cla()
    
    # #plt.show()
    
    
    # plt.imshow(v[0])
    # plt.show()
    # # left = left_edge(seq[0], 10)
    # # right = right_edge(seq[0], 10)
    # # top = top_edge(seq[0], 10)
    # # bottom = bottom_edge(seq[0], 10)

    # # time = np.arange(len(seq))
    # # for f in (left, right, top, bottom):
    # #     seq[np.ix_(time, f[0].flatten(), f[1].flatten())] = 0
    # from skimage.morphology import flood
    
    # red_seq[green_seq > red_seq] = 0

    # R = np.mean(red_seq, axis=0)
    # R[R < 0.6*np.mean(R)] = 0

    # X = ~np.array(R, dtype=bool)
    # keep_red = flood(X, (R.shape[0]//2, R.shape[1]//2))

    # space_red_mean = np.mean(red_seq, axis=(1, 2))
    # red_seq = red_seq / space_red_mean[:, np.newaxis, np.newaxis]
    
    # fred_seq = red_seq*keep_red
    
    # #G = np.mean(green_seq, axis=0)
    # #G[G < 0.55*np.mean(G)] = 0

    # #fig, ax = plt.subplots(1, 2)
    # #ax[0].imshow(R)
    # #ax[1].imshow(G)    
    # #plt.show()


    # #not_green = flood(G, (G.shape[0]//2, G.shape[1]//2))

    # #R_ = red_seq * not_green
    # #R_ = np.mean(R_, axis=0)

    # import os
    # not os.path.isdir('test') and os.mkdir('test')

    # red_seq = io.imread(rpath)
    
    # # fig, ax = plt.subplots(1, 2)
    # # for i in range(len(red_seq)):
    # #    ax[0].imshow(red_seq[i])
    # #    ax[0].axis('off')
       
    # #    ax[1].imshow(fred_seq[i])
    # #    ax[1].axis('off')       
    # #    fig.savefig('test/img_{:04d}.png'.format(i))
    # #    plt.cla()
    
    # #plt.show()
