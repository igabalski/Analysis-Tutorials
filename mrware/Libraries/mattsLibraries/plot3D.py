from plotStyles import *
from mathOperations import *

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

################################################################################
#~~~~~~~~~3D arrows
################################################################################
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


################################################################################
#~~~~~~~~~Plot arrows
################################################################################
def plotArrow(x,y,z,label=None, offset=0.1):
    ax = plt.gca()
    a = Arrow3D(x,y,z,mutation_scale=5,
                    lw=1, arrowstyle="-|>", color="k")
    if label is not None:
        ax.text(x[1]+offset,y[1]+offset,z[1]+offset, label  )
    ax.add_artist(a)

################################################################################
#~~~~~~~~~3D shapes
################################################################################
def plotSphere(x0=0,y0=0,z0=0,r=1,ntheta=30,nphi=30,clr='red',alpha=1):
    ax = plt.gca()

    u = np.linspace(0, np.pi, ntheta)
    v = np.linspace(0, 2 * np.pi, nphi)

    x = np.outer(np.sin(u), np.sin(v))*r
    y = np.outer(np.sin(u), np.cos(v))*r
    z = np.outer(np.cos(u), np.ones_like(v))*r

    # ax.plot_wireframe(x, y, z)
#     ax.plot_surface(x+x0, y+y0, z+z0, color=clr, alpha=.5)
    ax.plot_surface(x+x0, y+y0, z+z0, color=clr, alpha=alpha)

def plotCylinder(x0=0,y0=0,z0=0,r=1,h=10,ntheta=30,nz=30):
    ax = plt.gca()

    u = np.linspace(0, 2*np.pi, ntheta)
    z =  np.linspace(0, h, nz)

    UU,ZZ = np.meshgrid(u,z)

    XX = np.cos(UU)*r
    YY = np.sin(UU)*r

    # ax.plot_wireframe(x, y, z)
    ax.plot_surface(XX+x0, YY+y0, ZZ+z0)

################################################################################
#~~~~~~~~~Generate axes
################################################################################
def revArrow( zlims , rev=True ):
    if rev:
        return [-zlims[0], -zlims[1]]
    else:
        return zlims

def revLims( zlims , rev=True ):
    if rev:
        return [-zlims[1], -zlims[0]]
    else:
        return zlims

def plotXYZ(lmax=10, lmin=0, offset=None, sym=True, color='k', rev=[False,False,False], axlabel=['x','y','z']):
    ax = plt.gca()
    if sym:
        myRange = [-lmax,lmax]
    else:
        myRange = [lmin,lmax]

    if offset is None:
        offset = lmax/100.

    limKeys = [ 'xLims', 'yLims', 'zLims' ]
    lims = { 'xLims':[], 'yLims':[], 'zLims':[] }

    for idx in xrange(3):
        data = [[0,0] for x in xrange(3)]
        data[idx] = revArrow(myRange, rev[idx])
        lims[ limKeys[idx] ] = revLims(myRange, rev[idx])
        x,y,z = data
        a = Arrow3D(x,y,z,mutation_scale=5,
                    lw=1, arrowstyle="-|>", color=color, zorder=-100)
        ax.add_artist(a)

        revVal = 1.0* int(rev[idx] != True) - 1.0* int(rev[idx])
        # revSigns = [revVal,revVal,revVal]

        ax.text(x[1]+revVal*offset,
                y[1]+revVal*offset,
                z[1]+revVal*offset,
                axlabel[idx], color=color  )

    return lims


################################################################################
#~~~~~~~~~Molecule plotting
################################################################################
class molecule():
    def __init__(self,moleculeDictionary,bonds,atom_radius=.3):
        self.atoms     = moleculeDictionary.keys()
        self.positions = {key:np.array(moleculeDictionary[key]) for key in self.atoms}
        self.bonds     = bonds
        self.atom_colors = {'C':'grey','H':'white','O':'red','I':'#8A2BE2'}
        self.atom_radius = .3

    def drawMolecule(self):
        ax = plt.gca()
        for (A,B) in self.bonds:
#             x0, y0, z0 = self.positions[A]
#             x1, y1, z1 = self.positions[B]
            er = np.array( self.positions[B] ) - np.array( self.positions[A] )
            er = er / np.linalg.norm(er)

            start_at = np.array( self.positions[A] ) + er*self.atom_radius
            end_at  = np.array( self.positions[B] ) - er*self.atom_radius

            x0, y0, z0 = start_at
            x1, y1, z1 = end_at

            ax.plot([x0,x1], [y0,y1], [z0,z1],color='k', linewidth=2.5,zorder=-1)
        for A in self.atoms:
            x,y,z = self.positions[A]
            plotSphere(x,y,z,r=self.atom_radius,ntheta=60,nphi=60,clr=self.atom_colors[A[0]])

################################################################################
#~~~~~~~~~3D figure frame
################################################################################
def figure3D( xIn=3, yIn=3, dpi=300, axes_on=False, camera_at=None ):
    fig = plt.figure(figsize=(xIn, yIn), dpi=dpi)
    ax = plt.axes(projection='3d')
    ax._axis3don = axes_on
    if camera_at is not None:
        elev = camera_at[0]
        azim = camera_at[1]
        ax.view_init(elev=elev, azim=azim)
    return ax


################################################################################
#~~~~~~~~~3D axes limits
################################################################################
def generateBufferedLims( lims ):
    lims = np.array(lims)
    limBuffer = np.abs(lims).max()
    return np.array( [ lims[0]-0.1*limBuffer, lims[1]+0.1*limBuffer ] )

def setFigure3DAxesLimits( xLims=None, yLims=None, zLims=None, symLims=None ):
    if symLims is not None:
        xLims=symLims
        yLims=symLims
        zLims=symLims
    elif xLims is None:
        raise ValueError('Either symLims or (xLims,yLims,zLims) must be defined')

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlim( generateBufferedLims( xLims ) )
    plt.ylim( generateBufferedLims( yLims ) )
    ax.set_zlim( zLims )



################################################################################
#~~~~~~~~~3D color-plot
################################################################################

def colorPlot3D(X,Y,Z, zdir='z', zoffset=0, nContours=10, divergent=False, scale=1 ):
    vmin = Z.min()*scale
    vmax = Z.max()*scale

    if divergent:
        vmax = np.abs(Z).max()*scale



    ax=plt.gca()
    if divergent:
        ax.contourf(X,Y,Z, nContours,
                    zdir=zdir, offset=zoffset,
                    cmap=fireIce_cmap, vmin=-vmax, vmax=vmax)

        m = plt.cm.ScalarMappable(cmap=fireIce_cmap)
        m.set_array(Z)
        m.set_clim(-vmax, vmax)
    else:
        ax.contourf(X,Y,Z, nContours,
                    zdir=zdir, offset=zoffset,
                    cmap=fire_cmap, vmin=0, vmax=vmax)
        m = plt.cm.ScalarMappable(cmap=fire_cmap)
        m.set_array(Z)
        m.set_clim(0, vmax)
# ax.contourf(X,Z,Y, zdir='y', offset=10, cmap='RdBu_r')
