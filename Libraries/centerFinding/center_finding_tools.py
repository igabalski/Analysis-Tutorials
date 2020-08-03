from __future__ import division
import numpy as np


############################################################################
# downbin 1024 x 1024 image to speed up hough transform
############################################################################
def downbin(opal, d):
    if d == 1: im_opal = np.copy(opal)
    else:
        im_opal = np.copy(opal[:(1-d):d,:(1-d):d])
        for ind in range(1,d):
            im_opal += np.roll(opal,ind)[:(1-d):d,:(1-d):d]
    ax = np.arange(0, opal.shape[0], d)
    return im_opal, ax


############################################################################
# find peak of hough transformed image, can be replaced with better algorithm
############################################################################
def find_peak(image_in, p_of_max=0.9, edge_size=5):
    '''
    simple algorithm for finding peak of transformed image.
    zero out problem edges and threshold all values except peak,
    find centroid of remaining pixels.
    '''
    
    # initialize
    image = np.copy(image_in)
    sz = image.shape
    
    # threshold
    image[:edge_size], image[-edge_size:], image[:,:edge_size], image[:,-edge_size:] = 0, 0, 0, 0
    image[image < image.max()*p_of_max] = 0
    
    # find centroid of peak
    x = np.sum( np.arange(sz[0]) * np.sum(image,0) ) / np.sum(image)
    y = np.sum( np.arange(sz[1]) * np.sum(image,1) ) / np.sum(image)
    
    return int(x), int(y)


############################################################################
# create hough circle transformed image
############################################################################
def hough_circle_trans(image, radius_in, d, thresh=0, update=True):
    '''
    creates hough circle transformed image.
    maps circle of given radius to one point.
    does so by replacing each pixel with a circle of the given radius,
    then a peakfinder can find the circle's center.
    speed up by giving downbinned image or increasing thresh.
    '''
    
    # initialize size of original image, hough transformed image, circle image, and circumference of circle in pixels
    radius = int(radius_in / d)              # fix radius for downbin
    sz  = image.shape                        # original image size
    hsz = sz[0] + 2*radius, sz[1] + 2*radius # hough transformed image size
    csz = sz[0] + hsz[0], sz[1] + hsz[1]     # image of circle size
    sth = int(2*np.pi*radius)                # number of pixels in circle
    circle = np.zeros(csz)
    
    # create circle to be used for each pixel
    for theta in range(sth):
        xc = int( csz[0]/2 - radius * np.cos(theta*2*np.pi/sth) )
        yc = int( csz[1]/2 - radius * np.sin(theta*2*np.pi/sth) )
        circle[xc, yc] = 1
    if update: print 'circle created'
    
    # compute hough transform
    hough = np.zeros(hsz)
    for x in range(sz[0]):
        if (np.mod(x+1,100) == 0) and update: print '%d / %d' %(x+1, sz[0])
        for y in range(sz[1]):
            if image[x,y] <= thresh: continue
            hough += image[x,y] * circle[(sz[0] - x):(csz[0] - x), (sz[1] - y):(csz[1] - y)]
            
    print 'done with radius %d' %(radius*d)
    return hough