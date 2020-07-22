from __future__ import division
import numpy as np
import sys
sys.path.insert(0, '/reg/neh/home5/jneal/Libraries/image_manipulation/')
from aggregate import aggregate

cimport numpy as np
cimport cython


############################################################################
# make image out of hits
############################################################################
def cy_make_images(exc, eyc, ezc=None, data_filter=None):
    cdef np.ndarray[np.int16_t, ndim=1] xs
    cdef np.ndarray[np.int16_t, ndim=1] ys
    cdef np.ndarray[np.int16_t, ndim=2] im
    cdef np.ndarray[np.float32_t, ndim=1] zs
    
    cdef int i
    
    # check filter
    if data_filter is None:
        data_filter = np.ones_like(exc).astype(bool)
        
    # concatenate hits if necessary
    if np.isscalar(exc[0]):
        xs, ys = np.copy(exc.astype(np.int16)), np.copy(eyc.astype(np.int16))
        if ezc is not None: zs = np.copy(ezc.astype(np.float32))
    else:
        xs = np.concatenate(exc[data_filter])
        ys = np.concatenate(eyc[data_filter])
        if ezc is not None: zs = np.concatenate(ezc[data_filter])
    
    # stack into image
    im = aggregate(np.vstack((xs, ys)), 1, size=(1024,1024)).astype(np.int16)
    
    # if 3d, add height
    if ezc is not None:
        for i in range(ezc.size):
            im[xs,ys] *= zs
            
    return im, data_filter.sum()


############################################################################
# zero out ring on image
############################################################################
def cy_zero_mask(array, cen, int inradius, int outradius, inputType='im'):
    cdef np.ndarray[np.int16_t, ndim=1] x_in
    cdef np.ndarray[np.int16_t, ndim=1] y_in
    cdef np.ndarray[np.int16_t, ndim=2] mask
    
    cdef np.int16_t xc = cen[0]
    cdef np.int16_t yc = cen[1]
    cdef np.int16_t xn
    cdef np.int16_t yn
    
    cdef Py_ssize_t i
    
    if inputType == '3dhit': z = array[:,2].astype(np.float32)
    else: z = None
        
    # if given no center
    if cen is None:
        if inputType == 'im':
            out  = np.copy(array)
            x, y = None, None
        else:
            x, y  = array[:,0], array[:,1]
            out,_ = cy_make_images(x, y, z)
            
    # if given image
    elif inputType == 'im':
        xn, yn  = array.shape
        
        yy, xx  = np.ogrid[-xc:xn-xc, -yc:yn-yc]
        inmask  = xx**2 + yy**2 <= inradius**2
        outmask = xx**2 + yy**2 >= outradius**2
        
        out          = np.copy(array)
        out[inmask]  = 0
        out[outmask] = 0
        x, y         = None, None
    
    # if given array of hits
    else:
        x_in, y_in = array[:,0].astype(np.int16), array[:,1].astype(np.int16)
        x, y, zif  = [], [], []
        for i in range(x_in.shape[0]):
            r_check = (x_in[i] - xc)**2 + (y_in[i] - yc)**2
            
            if (r_check >= inradius**2) and (r_check <= outradius**2):
                x.append(x_in[i])
                y.append(y_in[i])
                if z is not None: zif.append(z[i])
                
        x, y  = np.array(x), np.array(y)
        if z is not None: z = np.array(zif)
        out,_ = cy_make_images(x, y, z)
        
    return out, x, y, z
        
        
############################################################################
# find peak of hough transformed 
# can be replaced with better algorithm
############################################################################
def cy_find_peak(np.ndarray[np.int32_t, ndim=2] im_in, perc_of_max=0.9, Py_ssize_t edge_size=5):
    '''
    simple algorithm for finding peak of transformed image.
    zero out problem edges and threshold all values except peak,
    find centroid of remaining pixels.
    '''
    cdef np.ndarray[np.int32_t, ndim=2] im = np.copy(im_in)
    
    cdef np.int16_t xw = im_in.astype(int).shape[0]
    cdef np.int16_t yw = im_in.astype(int).shape[1]
    cdef np.int16_t x
    cdef np.int16_t y
    
    cdef float im_sum
    
    # remove edges, threshold
    im[:edge_size], im[-edge_size:], im[:,:edge_size], im[:,-edge_size:] = 0, 0, 0, 0
    im[im < im.max()*perc_of_max] = 0
    
    # find centroid of peak
    if np.sum(im) == 0:
        print 'failed to find center'
        return 0, 0
    else:
        im_sum = np.sum(im)
        x = int(np.sum( np.arange(xw) * np.sum(im, 0) ) / im_sum)
        y = int(np.sum( np.arange(yw) * np.sum(im, 1) ) / im_sum)
        
        return x, y
    
    
############################################################################
# make circle
############################################################################
def cy_make_circle(int radius_in, int d, int thickness, csz=0):
    cdef np.ndarray[np.int16_t, ndim=2] circle
    
    cdef np.int16_t r     = np.copy(radius_in).astype(np.int16)
    cdef np.int16_t cen
    cdef np.int16_t a
    cdef np.int16_t cw
    
    cdef np.float32_t sth   = int(2*np.pi*r)
    cdef np.float32_t theta
    cdef np.float32_t t
    cdef np.float32_t sig = thickness / 3
    
    cdef Py_ssize_t xc
    cdef Py_ssize_t yc
    
    # initialize size of circle array
    if csz == 0:
        cw     = 2*r+1
        circle = np.zeros((cw, cw)).astype(np.int16)
        cen    = np.copy(r)
    else:
        cw     = np.copy(csz)
        circle = np.zeros((csz, csz)).astype(np.int16)
        cen    = int(csz / 2)
        
    # draw circle as gaussian ring
    for theta in range(int(sth)):
        for t in range(thickness):
            a = ( 100 * np.exp(-t**2 / sig**2 / 2) ).astype(np.int16)
            
            xc = ( cen - (r-t-thickness) * np.cos(theta*2*np.pi/sth) ).astype(np.int16)
            yc = ( cen - (r-t-thickness) * np.sin(theta*2*np.pi/sth) ).astype(np.int16)
            if xc is None: continue
            circle[xc, yc] = a
            
            xc = ( cen - (r+t-thickness) * np.cos(theta*2*np.pi/sth) ).astype(np.int16)
            yc = ( cen - (r+t-thickness) * np.sin(theta*2*np.pi/sth) ).astype(np.int16)
            if xc is None: continue
            circle[xc, yc] = a
            
    return circle, cw
            

############################################################################
# create hough circle transformed image
############################################################################
#@cython.boundscheck(False)
#@cython.wraparound(False)
def cy_hough_circle_trans(im_in, int radius_in, int d, region=None, x=None, y=None, thresh=0, int thickness=1, update=True):
    '''
    creates hough circle transformed image.
    maps circle of given radius to one point.
    does so by replacing each pixel with a circle of the given radius,
    then a peakfinder can find the circle's center.
    speed up by giving downbinned image or increasing thresh.
    '''
    if radius_in is None: print('radius_in is None')
        
    cdef np.ndarray[np.int16_t, ndim=2] im
    cdef np.ndarray[np.int16_t, ndim=2] circle
    cdef np.ndarray[np.int32_t, ndim=2] hough
    cdef np.ndarray[np.int16_t, ndim=1] x_in
    cdef np.ndarray[np.int16_t, ndim=1] y_in
    im = np.copy(im_in.astype(np.int16))
    
    cdef np.int16_t weight
    
    cdef Py_ssize_t r   = int(radius_in / d)
    cdef Py_ssize_t sz  = im.shape[0]
    cdef Py_ssize_t hsz
    cdef Py_ssize_t csz
    cdef Py_ssize_t w
    cdef Py_ssize_t i
    cdef Py_ssize_t xi
    cdef Py_ssize_t yi
    cdef Py_ssize_t xh
    cdef Py_ssize_t yh
    cdef Py_ssize_t xlh
    cdef Py_ssize_t ylh
    cdef Py_ssize_t xrh
    cdef Py_ssize_t yrh
    cdef Py_ssize_t xclh
    cdef Py_ssize_t yclh
    cdef Py_ssize_t xcrh
    cdef Py_ssize_t ycrh
    cdef Py_ssize_t xhlc
    cdef Py_ssize_t yhlc
    cdef Py_ssize_t xhrc
    cdef Py_ssize_t yhrc
    
    # setup image sizes ################################################################
    if region is None:
        hsz = sz + 2*r
        csz = sz + hsz
    else:
        hsz = 0
        csz = 0
    
    circle, cw = cy_make_circle(r, d, int(thickness/d), csz)
    if update: print('circle created')
        
    # no selected region ###############################################################
    if region is None:
        hough = np.zeros((hsz,hsz)).astype(np.int32)
        hax   = np.arange(-r,sz+r,d)
        # image
        if x is None:
            # loop through pixels, add circle to hough weighted by pixel value
            for xi in range(sz):
                if (np.mod(xi+1,100) == 0) and update: print '%d / %d' %(xi+1, sz)
                for yi in range(sz):
                    weight = im[xi,yi]
                    if weight <= thresh: continue
                    
                    hough += (weight * circle[(sz - xi):(csz - xi), (sz - yi):(csz - yi)]).astype(np.int32)
        
        # list of hits
        else:
            # loop through hits, add circle to hough
            x_in, y_in = (x/d).astype(np.int16), (y/d).astype(np.int16)
            for i in range(x.size):
                xi, yi = x_in[i], y_in[i]
                if (np.mod(i+1,100) == 0) and update: print '%d / %d' %(i+1, x.size)
                
                hough += circle[(sz - xi):(csz - xi), (sz - yi):(csz - yi)].astype(np.int32)
        
    # only transform over given region #################################################
    else:
        # setup hough image size
        xlh, xrh = int(region[0]/d), int(region[1]/d)
        ylh, yrh = int(region[2]/d), int(region[3]/d)
        xh,  yh  = xrh - xlh, yrh - ylh
        
        hough = np.zeros((xh,yh)).astype(np.int32)
        hax   = np.arange(xlh, xrh, d)
        
        # image
        if x is None:
            # loop through pixels, add circle to hough weighted by pixel value over narrow region
            for xi in range(sz):
                if (np.mod(xi+1,100) == 0) and update: print '%d / %d' %(xi+1, sz)
                for yi in range(sz):
                    weight = im[xi,yi]
                    if weight <= thresh: continue
                        
                    # find overlap of hough image and circle
                    xhlc, yhlc = -xlh + xi - r, -ylh + yi - r
                    xhrc, yhrc = -xlh + xi + r, -ylh + yi + r
                    xclh, yclh = r - xi + xlh,  r - yi + ylh
                    xcrh, ycrh = r - xi + xrh,  r - yi + yrh
                    
                    if (xhlc>=xh ) or (yhlc>=yh ) or (xhrc<=0) or (yhrc<=0): continue
                    if (xclh>=2*r) or (yclh>=2*r) or (xcrh<=0) or (ycrh<=0): continue
                    
                    if xhlc < 0:   xhlc = 0
                    if yhlc < 0:   yhlc = 0
                    if xhrc > xh:  xhrc = xh
                    if yhrc > yh:  yhrc = yh
                    if xclh < 0:   xclh = 0
                    if yclh < 0:   yclh = 0
                    if xcrh > 2*r: xcrh = 2*r
                    if ycrh > 2*r: ycrh = 2*r
                    
                    hough[xhlc:xhrc, yhlc:yhrc] += (weight * circle[xclh:xcrh, yclh:ycrh]).astype(np.int32)
                    
        # list of hits
        else:
            # loop through hits, add circle to hough over narrow region
            x_in, y_in = (x/d).astype(np.int16), (y/d).astype(np.int16)
            for i in range(x.size):
                xi, yi = x_in[i], y_in[i]
                if (np.mod(i+1,100) == 0) and update: print '%d / %d' %(i+1, x.size)
                
                # find overlap of hough image and circle
                xhlc, yhlc = -xlh + xi - r, -ylh + yi - r
                xhrc, yhrc = -xlh + xi + r, -ylh + yi + r
                xclh, yclh = r - xi + xlh,  r - yi + ylh
                xcrh, ycrh = r - xi + xrh,  r - yi + yrh

                if (xhlc>=xh ) or (yhlc>=yh ) or (xhrc<=0) or (yhrc<=0): continue
                if (xclh>=2*r) or (yclh>=2*r) or (xcrh<=0) or (ycrh<=0): continue

                if xhlc < 0:   xhlc = 0
                if yhlc < 0:   yhlc = 0
                if xhrc > xh:  xhrc = xh
                if yhrc > yh:  yhrc = yh
                if xclh < 0:   xclh = 0
                if yclh < 0:   yclh = 0
                if xcrh > 2*r: xcrh = 2*r
                if ycrh > 2*r: ycrh = 2*r
                
                hough[xhlc:xhrc, yhlc:yhrc] += circle[xclh:xcrh, yclh:ycrh].astype(np.int32)
                    
    print('done with radius %d' %(r*d))

    return hough, circle, hax