from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

import os
import sys
os.environ['folderPath'] = '/reg/neh/home/jneal'
sys.path.insert(0, os.environ['folderPath'] + '/mattsTools_master')
sys.path.insert(0, os.environ['folderPath'] + '/Jordan_LCLS_scripts/scripts')
sys.path.insert(0, os.environ['folderPath'] + '/Jordan_LCLS_scripts/scripts/electrons')

from surf import surf
from plotStyles import *
from cy_cen_tools import *

from cv2 import resize

cimport numpy as np
cimport cython

from time import time


'''
two input options:
 1) data_in is a square 2d numpy array
 2) data_in is a list of x, y positions of hits
 
center_find returns the downbinned image, center location, hough transformed image, circle, and axes
plot_center   plots image + center and image + center + circle
plot_im_hough plots image + circle and hough transformed image
'''
############################################################################
# return center found for given radius, reads in image
# wrapper for center finding
############################################################################
def cy_cen_find(data_in, radii, int d, # downbinning factor
                r_sum=None,      # radii to sum together after hough circle transform
                im_thresh=1,     # set all pixels below to zero, speeds up hough circle transform
                pk_thresh=0.95,  # sets any pixel below pk_thresh * max to zero when peakfinding the transformed image
                int thickness=1, # radius of thickness of ring in hough transform
                int edge_size=5, # how far into the edge of the transformed image to set to zero
                update=True,     # give detailed updates when computing transform?
                region=None,     # used to speed up performance by selecting region of expected center
                cen_guess=None,  # used to zero out center of image
                int inrad=100,   # radius of center zeroed out
                int outrad=900,  # radius of outer ring zeroed out
                inputType='im',  # tells if input is an image or array of hits ('im' or 'hit' or '3dhit')
                outputname='center_im',
                filepath='/processed/electrons/detector_images/',
                install_path='/reg/neh/home5/jneal/Jordan_LCLS_scripts'):
    
    # initialize
    hough_im, circle_im, hax, xpk, ypk = [], [], [], [], []
    
    # zero out center and outside ring, fill im, x, and y
    im, x, y, z = cy_zero_mask(data_in, cen_guess, inrad, outrad, inputType)
    
    cdef np.ndarray[np.int16_t, ndim=1] dx
    
    cdef np.int16_t sz = im.shape[0]
    cdef np.int16_t sd = int(sz/d)
    cdef np.int16_t ind
    cdef np.int16_t rd
    cdef np.int16_t r
                    
    if (sz != im.shape[1]) or (sz != 1024): print 'wrong image shape'
    
    # downbin image
    im_d = resize(im.astype(float),(sd,sd))
    dx   = np.arange(0,sz,d).astype(np.int16)
    
    # compute hough transform, find center
    for ind, r in enumerate(radii):
        hough_temp, circle_temp, hax_temp = cy_hough_circle_trans(im_d, r, d, region, x, y, im_thresh, thickness, update)
        
        hough_im.append(hough_temp)
        circle_im.append(circle_temp)
        hax.append(hax_temp)
        
        pk = cy_find_peak(hough_temp, pk_thresh, edge_size)
        
        if region is None:
            xpk.append( d*pk[0] - r )
            ypk.append( d*pk[1] - r )
        else:
            xpk.append( d*pk[0] + region[0] )
            ypk.append( d*pk[1] + region[2] )

    print 'chosen radius, x, y center pixel'
    print np.array(radii)
    print np.array(xpk)
    print np.array(ypk)
    
    # stack given ransformed images, find center of stacked image
    if (r_sum is not None) and (region is None):
        h_tot = np.zeros_like(im_d)
        for ind, r in enumerate(r_sum):
            if r in radii:
                rd = int(r/d)
                h_tot += hough_im[ind][rd:(sd+rd),rd:(sd+rd)]
        pk_tot = d * np.array( cy_find_peak(h_tot, pk_thresh, edge_size) ) - r
    else:
        pk_tot = np.array([xpk[0],ypk[0]])

    # save center
    center_best = np.array(pk_tot)
    if outputname is not None:
        np.save( install_path + filepath + 'center_' + outputname, center_best )
        
    return im_d, xpk, ypk, pk_tot, hough_im, circle_im, dx, hax


############################################################################
# compute streaking amount and angle
############################################################################
def compute_streaking(cen_avg, cen):
    xdiff     = cen[0] - cen_avg[0]
    ydiff     = cen[1] - cen_avg[1]
    streaking = np.sqrt( (xdiff)**2 + (ydiff)**2 )
    angle     = np.arctan2( ydiff, xdiff ) * 180 / np.pi
    
    return streaking, angle


############################################################################
# plot center and circle over original image
############################################################################
def plot_center(im_d, d, radii, xpk, ypk, pk_tot, dx, hax, zmax = 15):
    sz = im_d.shape[0]
    if sz != im_d.shape[1]: print('wrong image shape')
    
    figOpts = { 'xIn':6, 'yIn':5, 'xLims':[0,1024], 'yLims':[0,1024], 'zLims':[0,zmax] }
    
    colorPlot(dx, dx, im_d, **figOpts)
    plt.scatter( xpk,       ypk,       s=10, facecolors='None', edgecolors='w')
    plt.scatter( pk_tot[0], pk_tot[1], s=10, facecolors='None', edgecolors='r')
    
    colorPlot(dx, dx, im_d, **figOpts)
    ax=plt.gca()
    ax.add_patch(plt.Circle((pk_tot[0], pk_tot[1]), radii[0], facecolor='None', edgecolor='w', linestyle=':'))
    plt.scatter( xpk,       ypk,       s=10, facecolors='None', edgecolors='w')
    plt.scatter( pk_tot[0], pk_tot[1], s=10, facecolors='None', edgecolors='r')
    plt.show()
    
    
############################################################################
# plot center and circle over original image next to hough transformed image
############################################################################
def plot_im_hough(im_d, d, radii, xpk, ypk, pk_tot, hough_im, circle_im, dx, hax, zmax = 15):
    sz = im_d.shape[0]
    
    for ind, r in enumerate(np.array(radii)):
        edge = int(r/d) + sz
        x, y = int(xpk[ind]/d), int(ypk[ind]/d)
        
        plt.figure(figsize=(20,7))
        plt.subplot(1,2,1)
        try:
            surf(dx, dx, im_d + zmax*circle_im[0][(edge-x):( sz+edge-x ), (edge-y):( sz+edge-y )].T)
        except:
            print('failed to plot im + circle')
        plt.scatter(xpk[ind], ypk[ind], s=200, facecolors='None', edgecolors='w')
        plt.xlim((0,sz*d))
        plt.ylim((0,sz*d))
        plt.clim((0,zmax))
        plt.title('for radius %d, peak at (%d, %d)' %(r, xpk[ind], ypk[ind]))
        
        plt.subplot(1,2,2)
        if hax[ind].shape[0] == hough_im[ind].shape[0]:
            surf(hax[ind], hax[ind], hough_im[ind])
        else:
            surf(np.arange(hough_im[ind].shape[0]), np.arange(hough_im[ind].shape[1]), hough_im[ind])
        plt.scatter(xpk[ind], ypk[ind], s=200, facecolors='none', edgecolors='w')
        plt.xlim((-r,sz*d+r))
        plt.ylim((-r,sz*d+r))