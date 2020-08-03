import numpy as np
from scipy.interpolate import interp1d
from pyTools import *

################################################################################
#~~~~~~~~~Log ops
################################################################################
def logPolyVal(p,x):
    ord = p.order()
    logs = []
    for idx in xrange(ord+1):
        logs.append( np.log( p[idx] ) + (ord-idx)*np.log(x) )
    return logs

################################################################################
#~~~~~~~~~Symmeterize data
################################################################################
def symmeterize( x, y, interp_type='cubic' ):
    if x.min() <= 0:
        raise ValueError('x.min() must be greater than zero.')

    xs = np.array([-x,x]).flatten()
    xs.sort()

    f = interp1d( x , y , kind=interp_type )

    return { 'x':xs , 'y':f(np.abs(xs)) }

################################################################################
#~~~~~~~~~3D Shapes
################################################################################
def makeSphere(x0=0,y0=0,z0=0,r=1,ntheta=30,nphi=30):
    u = np.linspace(0, np.pi, ntheta)
    v = np.linspace(0, 2 * np.pi, nphi)

    x = np.outer(np.sin(u), np.sin(v))*r
    y = np.outer(np.sin(u), np.cos(v))*r
    z = np.outer(np.cos(u), np.ones_like(v))*r
    return x+x0, y+y0, z+z0

def makeCylinder(x0=0,y0=0,z0=0,r=1,h=10,ntheta=30,nz=30):
    u = np.linspace(0, 2*np.pi, ntheta)
    z =  np.linspace(0, h, nz)

    UU,ZZ = np.meshgrid(u,z)

    XX = np.cos(UU)*r
    YY = np.sin(UU)*r

    # ax.plot_wireframe(x, y, z)
    return XX+x0, YY+y0, ZZ+z0

def generateLine3D( x0=0, x1=1, y0=0, y1=1, z0=0, z1=0, N=2 ):
    return {'line':{'xData':np.linspace(x0,x1,N),
            'yData':np.linspace(y0,y1,N),
            'zData':np.linspace(z0,z1,N),
            'cData':np.ones((N,1))}}

################################################################################
#~~~~~~~~~2D Shapes
################################################################################
def generateCircle(R=1, X0=0, Y0=0, N = 60, thetaMin = 0, thetaMax = 2*np.pi ):
    thetas = np.linspace( thetaMin , thetaMax , N)
    uY = np.sin( thetas )*R
    uX = np.cos( thetas )*R
    return {'circle':{'xData':uX+X0, 'yData':uY+Y0}}

def generateEllipse( RX=2, RY=1, X0=0, Y0=0, N = 60, thetaMin = 0, thetaMax = 2*np.pi ):
    thetas = np.linspace( thetaMin , thetaMax , N)
    uY = np.sin( thetas )*RY
    uX = np.cos( thetas )*RX
    return {'ellipse':{'xData':uX+X0, 'yData':uY+Y0}}

def makeCylinder2D( L = 10., R = 1., N=60, view_degrees=30. ):

    yFac = np.cos(view_degrees * np.pi/180.)
    zFac = np.sin(view_degrees * np.pi/180.)

    xL = np.ones((2,1))*-R
    xR = -xL
    y  = np.array([0,L])*yFac

    cylinder = { 'leftSide':{'xData':xL, 'yData':y},
                 'rightSide':{'xData':xR, 'yData':y},
                 'upperEllipse':generateEllipse(RX = R, RY=R*zFac, Y0=L*yFac,N=N)['ellipse'],
                 'lowerHalfEllipse':generateEllipse(RX = R, RY=R*zFac, thetaMin=np.pi, thetaMax=2*np.pi, N=int(N/2.))['ellipse']}
    return cylinder

################################################################################
#~~~~~~~~~Rotations
################################################################################
def rotateObject(x,y,z,ax=None,ay=None,az=None):
    if ax is not None:
        y,z = rotateAt(y,z,ax)
    if ay is not None:
        x,z = rotateAt(x,z,-ay)
    if az is not None:
        x,y = rotateAt(x,y,az)
    return x,y,z

def rotateAt(x,y,a):
    xp = np.cos(a)*x-np.sin(a)*y
    yp = np.cos(a)*y+np.sin(a)*x
    return xp, yp

def rotateObj2D( obj_in, degrees ):
    obj = obj_in.copy()

    keys = obj.keys()
    for key in keys:
        obj[key] = rotate2D( degrees=degrees, **obj[key] )

    return obj

def rotate2D( xData, yData, degrees ):
    x = xData.flatten()
    y = yData.flatten()
    z = np.zeros_like(x)
    x,y,z = rotateObject( x, y, z, az=float(degrees)/180.*np.pi )
    return {'xData':x, 'yData':y}


def rotateObj3D( obj_in, gamma, theta, phi ):
    obj = obj_in.copy()

    keys = obj.keys()
    for key in keys:
        obj[key] = rotate3D( gamma=gamma, theta=theta, phi=phi, **obj[key] )

    return obj

def rotate3D( xData, yData, zData, gamma, theta, phi, kwargs_toggle=True, **kwargs ):
    ignore_kwargs(kwargs, toggle=kwargs_toggle)
    x = xData.flatten()
    y = yData.flatten()
    z = zData.flatten()

    x,y,z = rotateObject( x, y, z, az=float(gamma)/180.*np.pi )
    x,y,z = rotateObject( x, y, z, ay=float(theta)/180.*np.pi )
    x,y,z = rotateObject( x, y, z, az=float(phi)/180.*np.pi )

    return {'xData':x, 'yData':y, 'zData':z}
