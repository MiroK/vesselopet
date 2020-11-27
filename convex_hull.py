import collections
import numpy as np


Point = collections.namedtuple('Point', ('x', 'y'))

def convex_hull(points):
    "Find the convex hull of a set of points."
    # This is Peter Norvig's algorithm    
    if len(points) <= 3:
        return points
    # Find the two half-hulls and append them, but don't repeat first and last points
    upper = half_hull(sorted(points))
    lower = half_hull(reversed(sorted(points)))
    return upper + lower[1:-1]


def half_hull(sorted_points):
    "Return the half-hull from following points in sorted order."
    # Add each point C in order; remove previous point B if A->B-C is not a left turn.
    hull = []
    for C in sorted_points:
        # if A->B->C is not a left turn ...
        while len(hull) >= 2 and turn(hull[-2], hull[-1], C) != 'left':
            hull.pop() # ... then remove B from hull.
        hull.append(C)
    return hull


def turn(A, B, C):
    "Is the turn from A->B->C a 'right', 'left', or 'straight' turn?"
    diff = (B.x - A.x) * (C.y - B.y)  -  (B.y - A.y) * (C.x - B.x) 
    return ('right' if diff < 0 else
            'left'  if diff > 0 else
            'straight')


def mass_filter(array, tol, rel_mass=0.8, max_it=10, weighted=True):
    '''Hull of points array > tol.'''
    # The idea here is that if we use raw data that there is noise and
    # this makes convex hull way too big. Denoising is one option but
    # https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv/30380543
    # is very ad hoc for each image

    # So the idea here is this one: we are after a footprint of a flow
    # in an artery and we assume that most of the colored pixels are close
    # to the center of mass. We are fine leaveing behind some mass which
    # is far - this is where noise seems to be
    points = np.column_stack(np.where(array > tol))
    weights = np.array([array[tuple(p)] for p in points])
    if not weighted:
        weights *= 0
    
    total_mass = array.sum()
    # We weight be the pixel intensity
    center = sum((p*w for p, w in zip(points, weights)), np.zeros(2))
    center /= total_mass

    max_radius = np.max(np.linalg.norm(points - center, 2, axis=1))

    # We start the search from half way
    radius = 0.5*max_radius
    niters = 0
    while niters < max_it:
        niters += 1
        idx_inside,  = np.where(np.linalg.norm(points-center, 2, axis=1) < radius)
        mass_inside = sum(weights[idx_inside])
        if mass_inside > rel_mass*total_mass:
            break

        radius = radius + 0.5*(max_radius-radius)

    array_ = np.zeros_like(array)
    for i in idx_inside:
        array_[tuple(points[i])] = array[tuple(points[i])]

    return array_, center, max_radius


def array_convex_hull(array):
    idx0, idx1 = np.where(array > 0)
    return np.array(convex_hull([Point(*p) for p in zip(idx0, idx1)]))

    
def hull_center(hull):
    '''Non-interger center'''
    return np.mean(hull, axis=0)


def hull_area(hull, center=None):
    '''Triangulate and sum'''
    if center is None:
        center = hull_center(hull)

    tri_area = lambda A, B, C: 0.5*abs(np.cross(B-A, C-A))

    n = len(hull)
    area = 0.
    for i0 in range(n):
        i1 = (i0 + 1) % n
        area += tri_area(hull[i0], hull[i1], center)

    return area


def center_of_mass(img, tol=0):
    idx0, idx1 = np.where(img > tol)

    weights = img[idx0, idx1]
    points = np.c_[idx, idx1]

    mass = np.sum(weights)
    centers = np.sum(points*weights[:, np.newaxis], axis=0)

    return mass/centers

# --------------------------------------------------------------------

if __name__ == '__main__':

    A = np.zeros((10, 10))
    A[3, 3] = 5
    A[4, 3] = 5
    A[5, 3] = 5
    A[4, 4] = 5
    A[4, 2] = 5
    
    # x = array_convex_hull(A, tol=2)

    from array2gif import write_gif
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio
    #import cv2

    from sklearn import svm    
    from skimage.measure import EllipseModel
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    

    outliers_fraction = 0.2
    
    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=4, contamination=outliers_fraction))
    ]
    
    
    gif = [np.array(g[:, :, 0]) for g in imageio.mimread('data/red.gif')]


    theta = np.linspace(0, 2*np.pi, 100)

    ellipses = []
    for step, img in enumerate(gif[:]):
        ellipse = EllipseModel()

        d = 10
    
        shape = img.shape
        img[:, tuple(range(d))] = 0
        img[:, tuple(range(shape[0]-d, shape[0]))] = 0
        img[tuple(range(d)), :] = 0
        img[tuple(range(shape[0]-d, shape[0])), :] = 0

        X = np.array(np.where(img > 0)).T
        preds = {}
        for name, algorithm in anomaly_algorithms:
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)
                  
            preds[name] = y_pred
        #filtered, center, max_radius = mass_filter(img, tol=0, rel_mass=0.8, max_it=10,
        #                                           weighted=True)

        pts = np.array(np.where(img > 0)).T
        print(pts)
        # filtered, center1, max_radius1 = mass_filter(filtered, tol=0, rel_mass=0.8, max_it=10)
        #filtered, center2, max_radius2 = mass_filter(filtered, tol=0, rel_mass=0.8, max_it=10)        
        #x, y = array_convex_hull(filtered).T
        #x = np.r_[x, x[0]]
        #y = np.r_[y, y[0]]


        fig, axarr = plt.subplots(1, 1+len(anomaly_algorithms),
                                  figsize=(16, 10))
        plt.title('Step {}'.format(step))
        ax = axarr.ravel()
        
        ax[0].imshow(img)

        for i, y_pred in enumerate(preds.values(), 1):
            foo = np.zeros_like(img)
            for y, pt in zip(y_pred, pts):
                foo[tuple(pt)] = img[tuple(pt)] if y >  0 else 0
            #foo[np.ix_(pts[:, 0], pts[:, 1])] = y_pred

            ax[i].imshow(foo)

            x, y = array_convex_hull(foo).T
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
            # ax[1].imshow(filtered)
            ax[i].plot(y, x, marker='o')
        #ax[0].plot(y, x, marker='o')

        # markers = iter(['x', '+', 's'])
        # colors = iter(['orange', 'magenta', 'brown'])
        # for cc, rr in zip((center, ),#, center1),#, center2),
        #                   (max_radius, )):#, max_radius1)):#, max_radius2)):
        #     marker, color = next(markers), next(colors)
            
        #     xc, yc = cc
        #     x = xc + rr*np.sin(theta)
        #     y = yc + rr*np.cos(theta)
        #     ax[1].plot(y, x, color=color)
        #     ax[1].plot(yc, xc, marker=marker, color=color)
        #     #ax[1].set_xlim((0, 100))
        #     #ax[1].set_ylim((100, 0))

            status = ellipse.estimate(np.c_[x, y])

            if status:
                xc, yc, a, b, theta0 = ellipse.params

                x = xc + a*np.sin(theta + theta0)
                y = yc + b*np.cos(theta + theta0)

                ax[i].plot(y, x, color='red')
                ax[i].set_xlim((0, 99))
                ax[i].set_ylim((0, 99))
                ax[i].invert_yaxis()
            
        fig.savefig('cvx_hull_{:03d}.png'.format(step))

            # ellipses.append([xc, yc, a, b, theta0])

    # ellipses = np.array([ellipses])
    # xc, yc, a, b, theta0 = ellipses.T
    
    # plt.figure()
    # plt.plot(xc, label='xc')
    # plt.plot(yc, label='yx')
    # plt.legend()
    
    # fig, ax = plt.subplots()
    # plt.plot(a, label='major')
    # plt.plot(b, label='minor')

    #ax2 = ax.twinx()
    #ax2.set_ylabel('area', color='black')  # we already handled the x-label with ax1    
    #ax2.plot(np.pi*a*b, color='black')
    
    #plt.figure()
    #plt.plot(theta0, label='ellipse angle')
    #plt.legend()
    

    #plt.show()
    
# # If I start from the big convex hull and then consider the hull of the
# # points enclosed by previous hull - vertex which 2 longest vertices
# # what happens with volume vs circumnference?
# #
# # -conseq triangles
# # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html

