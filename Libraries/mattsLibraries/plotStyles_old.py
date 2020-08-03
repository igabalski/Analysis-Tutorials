import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from fireIce import *
from mathOperations import *
from pyTools import *

################################################################################
#~~~~~~~~~Default cycler
################################################################################
from cycler import cycler

def setLineCycler():
    ax = plt.gca()

    colors = ['k','k','k','k']
    linestyles = ['-','--',':','-.']

    mycycler = cycler('linestyle', linestyles) + cycler('color',colors)

    # plt.rc('axes',prop_cycle=mycycler)
    ax.set_prop_cycle( mycycler )

def resetLineCycler():
    setLineCycler()

################################################################################
#~~~~~~~~~Font properties
################################################################################
import matplotlib.font_manager as font_manager
matplotlib.rcParams.update({'font.size': 10})

matplotlib.rcParams['mathtext.fontset'] = "cm"
matplotlib.rcParams['font.serif'] = "CMU Serif"
# # Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"
# matplotlib.rcParams['font.family'] = prop.get_name()


################################################################################
#~~~~~~~~~Custom colormaps
################################################################################
import matplotlib.colors
WR_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])
KR_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red"])
BKR_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","black","red"])
RW_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white"])
fireIce_cmap = fireIce()
fire_cmap   = fire()

################################################################################
#~~~~~~~~~Default meshes
################################################################################
def blueRedMesh( X, Y, Z, xlabel='x', ylabel='y', vmax=None):
    absmax = np.abs(Z).max()
    if vmax is None:
        vmax = absmax

    plt.pcolormesh(X,Y,Z,
                   cmap='RdBu_r', vmin=-vmax ,vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    return cbar

def defaultMesh( X, Y, Z, xlabel='x', ylabel='y', vmin=None, vmax=None, generateCBAR=True, zOrder=None, plotOptions={}  ):
    vmin0, vmax0 = vmin, vmax

    if (not generateCBAR):
        vmin0, vmax0 = plt.gci().get_clim()

    if vmin0 is None:
        vmin0, vmax0 = vmin, vmax
    vmin, vmax = vmin0, vmax0
    print vmin,vmax
    # plt.pcolormesh(X,Y,Z,cmap='inferno',vmin=vmin, vmax=vmax,linewidth=0,rasterized=True)
    # plt.pcolormesh(X,Y,Z,cmap=KR_cmap,vmin=vmin, vmax=vmax,linewidth=0,rasterized=True)
    if zOrder is not None:
        plt.pcolormesh(X,Y,Z,cmap=fire_cmap,vmin=vmin, vmax=vmax,linewidth=0,rasterized=True, zorder=zOrder, **plotOptions)
    else:
        plt.pcolormesh(X,Y,Z,cmap=fire_cmap,vmin=vmin, vmax=vmax,linewidth=0,rasterized=True, **plotOptions)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if generateCBAR:
        cbar = plt.colorbar()
        return cbar
    return None


def divergentMesh( X, Y, Z, xlabel='x', ylabel='y', vmax=None, generateCBAR=True, zOrder=None, plotOptions={}  ):
    vmin0, vmax0 = -vmax, vmax

    absmax = np.abs(Z).max()
    if vmax is None:
        vmax = absmax
    if not generateCBAR:
        vmin0, vmax0 = plt.gci().get_clim()
    if vmin0 is None:
        vmin0, vmax0 = -vmax, vmax
    vmin, vmax = vmin0, vmax0
    print vmax
    # plt.pcolormesh(X,Y,Z,cmap='inferno',vmin=vmin, vmax=vmax,linewidth=0,rasterized=True)
    # plt.pcolormesh(X,Y,Z,cmap=BKR_cmap,vmin=-vmax, vmax=vmax,linewidth=0,rasterized=True)
    if zOrder is not None:
        plt.pcolormesh(X,Y,Z,cmap=fireIce_cmap,vmin=-vmax, vmax=vmax,linewidth=0,rasterized=True, zorder=zOrder, **plotOptions)
    else:
        plt.pcolormesh(X,Y,Z,cmap=fireIce_cmap,vmin=-vmax, vmax=vmax,linewidth=0,rasterized=True, **plotOptions)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if generateCBAR:
        cbar = plt.colorbar()
        return cbar
    return None



def fillBackground( figOpts, atZ = 0., zOrder=-100 ):
    xg = np.linspace( figOpts['xLims'][0],figOpts['xLims'][1],3 )
    yg = np.linspace( figOpts['yLims'][0],figOpts['yLims'][1],3 )

    [XX,YY] = np.meshgrid(xg,yg)

    colorPlot( XX, YY, XX*0, zOrder=zOrder, newFigure=False, **figOpts )

################################################################################
#~~~~~~~~~Return defaults
################################################################################
def defaultCycleColors():
    colorCycle = matplotlib.rcParams['axes.prop_cycle']
    cycleColors = []
    for c in colorCycle:
        cycleColors.append(c['color'])
    return cycleColors

################################################################################
#~~~~~~~~~Axis generators
################################################################################
def generateTickLabels( ticks ):
    n = getLargestOrder(ticks)
    return [printTexFloatFor(tick,n) for tick in ticks]

def generateAxisLabel( label, n, units ):
    if n is None:
        n = 0
    # if n == 1:
    #     return r'%s ($\times10$ %s)' % ( label, units )
    if (n==0)|(n==1):
        if (len(units)==0):
            return r'%s' % (label)
        else:
            return r'%s (%s)' % (label, units)
    else:
        return r'%s ($\times10^{%d}$ %s)' % ( label, n, units )

def generateTicks( data, nTicks, dMin = None, dMax = None ):
    if dMin is None:
        dMin = data.min()
    if dMax is None:
        dMax = data.max()

    ticks = np.linspace(dMin, dMax, nTicks)

    if ( dMin < 0 ) & ( dMax > 0 ):
        ticks = ticks - ticks[ np.abs(ticks).argmin() ]

    return ticks


################################################################################
#~~~~~~~~~Plot line plot
################################################################################
def linePlot( xData, yData,
              nxTicks=5, nyTicks=5,
              xUnits='', yUnits='',
              xLabel='$x$', yLabel='$y$',
              xLims=[None,None],
              yLims=[None,None],
              nxMinor=None, nyMinor=None,
              newFigure=True,
              xIn=3, yIn=2, dpi=300,
              plotOptions={},
              squareAxes=False,
              showAxes=True, mute_kwargs=False,
              bufferLims=False, **kwargs):

    if (kwargs is not None) and (not mute_kwargs):
        print 'Ignoring undefined input variable ...'
        print kwargs.keys()

    if squareAxes:
        xIn = yIn

    if newFigure:
        plt.figure(figsize=(xIn, yIn), dpi=dpi)
        setLineCycler()

    plt.plot(xData, yData, **plotOptions)
    generateAxes(xData, yData, nxTicks=nxTicks, nyTicks=nyTicks,
                 xUnits=xUnits, yUnits=yUnits, xLabel=xLabel, yLabel=yLabel,
                 xLims=xLims, yLims=yLims, nxMinor=nxMinor, nyMinor=nyMinor,
                 showAxes=showAxes, bufferLims=bufferLims)
    if squareAxes:
        plt.axis('equal')


################################################################################
#~~~~~~~~~Colorplot
################################################################################
def colorPlot( xData, yData, zData,
              nxTicks=5, nyTicks=5, nzTicks=5,
              xUnits='', yUnits='', zUnits='',
              xLabel='$x$', yLabel='$y$', zLabel='$z$',
              xLims=[None,None],
              yLims=[None,None],
              zLims=[None,None],
              divergent=False,
              newFigure=True,
              xIn=3, yIn=3, dpi=300,
              plotOptions={},
              showAxes=True, bufferLims=False,
              zOrder = None, generateCBAR = False):
    '''
        colorPlot generates a texified pcolormesh plot using scientific
        notation on the axes.

        Arguments:
        Required:
        xData, yData, and zData: 2D arrays specifying the x,y pixel positions and the color height, z.

        Optional:
        nxTick, nyTicks, nzTicks: Number of tick marks for the x,y, and color axes
        xUnits, yUnits, and zUnits: Unit of measurement for the x,y, and color axes
        xLabel, yLabel, and zLabel: Label for the x,y, and color axes
        xLims, yLims, and zLims: Limits for the x,y, and color axes

        divergent: False for positive definite data. True otherwise.
        newFigure: Set to False if you are combining with pre-existing plots.
        xIn, yIn: x and y inches for the current frame.
        dpi: Dots per square inch (300 for print quality)
        plotOptions: Dictionary that gets handed to the pcolormesh command as kwargs.
        showAxes: Turn the axes on or off
        bufferLims: If True, adds buffer 0.1*max(abs( axes_grid )) to the limits
        zOrder: If None, default ordering. If -100, will plot first.

        Returns:
        None, plots figure
    '''

    if newFigure:
        plt.figure(figsize=(xIn, yIn), dpi=dpi)

    if divergent:
        cbar = divergentMesh( xData, yData, zData, xlabel='x', ylabel='y', vmax=zLims[0], generateCBAR=(newFigure|generateCBAR), zOrder=zOrder, plotOptions=plotOptions  )

    else:
        cbar = defaultMesh( xData, yData, zData, xlabel='x', ylabel='y', vmin=zLims[0], vmax=zLims[1], generateCBAR=(newFigure|generateCBAR), zOrder=zOrder, plotOptions=plotOptions  )

    if newFigure:
        generateColorbar( cbar, zData, ncTicks=nzTicks, cUnits=zUnits, cLabel=zLabel, cLims=zLims )

    generateAxes(xData.flatten(), yData.flatten(), nxTicks=nxTicks, nyTicks=nyTicks,
                 xUnits=xUnits, yUnits=yUnits, xLabel=xLabel, yLabel=yLabel,
                 xLims=xLims, yLims=yLims, showAxes=showAxes, bufferLims=bufferLims)

################################################################################
#~~~~~~~~~Colorbar generation
################################################################################
def generateColorbar( cbar, cData, ncTicks=5, cUnits='', cLabel='z', cLims=[None,None] ):
    cTicks = generateTicks( cData, ncTicks, dMin=cLims[0], dMax=cLims[1] )
    cbar.set_ticks( cTicks )
    cbar.set_ticklabels( generateTickLabels( cTicks ) )
    ncMax = getLargestOrder( cTicks )

    plt.title( generateAxisLabel( cLabel , ncMax , cUnits ) )

    if cLims[0] is not None:
        plt.clim(np.array(cLims)*1.1)

################################################################################
#~~~~~~~~~Axes generation
################################################################################
def generateAxes(xData, yData,
              nxTicks=5, nyTicks=5,
              xUnits='', yUnits='',
              xLabel='x', yLabel='y',
              xLims=[None,None],
              yLims=[None,None],
              nyMinor=None,nxMinor=None,
              showAxes=True, bufferLims=False):
    # Get current axes
    ax = plt.gca()

    if showAxes:
        # Set the x axes
        xTicks = generateTicks( xData, nxTicks, dMin=xLims[0], dMax=xLims[1] )
        ax.set_xticks( xTicks )
        ax.set_xticklabels( generateTickLabels(xTicks) )


        # Set the y axes
        yTicks = generateTicks( yData, nyTicks, dMin=yLims[0], dMax=yLims[1] )
        ax.set_yticks( yTicks )
        ax.set_yticklabels( generateTickLabels(yTicks) )


        nxMax = getLargestOrder(xTicks)
        nyMax = getLargestOrder(yTicks)

        plt.xlabel( generateAxisLabel( xLabel, nxMax, xUnits ) )
        plt.ylabel( generateAxisLabel( yLabel, nyMax, yUnits ) )

        # Plot minor axes if nxMinor / nyMinor specified
        if nxMinor is not None:
            ax.xaxis.set_minor_locator( AutoMinorLocator( nxMinor ) )
        if nyMinor is not None:
            ax.yaxis.set_minor_locator( AutoMinorLocator( nyMinor ) )

    if xLims[0] is not None:
        absMax = np.abs(xLims).max()
        buffer = 0
        if bufferLims:
            buffer = 0.1*absMax
        plt.xlim([xLims[0]-buffer,xLims[1]+buffer])
    if yLims[0] is not None:
        absMax = np.abs(yLims).max()
        buffer = 0
        if bufferLims:
            buffer = 0.1*absMax
        plt.ylim([yLims[0]-buffer,yLims[1]+buffer])

    if not showAxes:
        # Hide the Axes
        plt.axis('off')

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

def savefig( name ):
    plt.savefig( name , bbox_inches='tight', transparent=True )

################################################################################
#~~~~~~~~~Texifying
################################################################################
def printFloat( f , d=2 ):
    return ('%.'+str(d)+'f')%f

def printTexFloat( f ):

    if f == 0.:
        return r'$0.0$'

    n = getOrder( f )
    if ( n > 1 ) | ( n < -1 ):
        front = ('%.1f' % (f/10**n))
        return r'$%s\times10^%d$' % (front,n)
    elif ( n == -1 ):
        return r'$%.2f$'%f
    elif ( n == 1 ):
        return r'$%.0f$'%f
    else:
        return r'$%.1f$'%f

def printTexFloatFor( f , n ):
    if f == 0.:
        return r'0.0'
    elif n == 1:
        if np.abs(f) == 10:
            front = (r'%.0f' % (f))+'.'
        else:
            front = (r'%.1f' % (f))
        return front
    else:
        front = (r'%.1f' % (f/10**n))
        return front

def getLargestOrder( floatList ):
    nmax = None
    for f in floatList:
        if f == 0:
            continue

        if nmax is None:
            nmax = getOrder(f)
        else:
            n = getOrder(f)
            if n > nmax:
                nmax = n
    return nmax


def getOrder( f ):
    n = 0
    f = np.abs(f)
    if ( f >= 10 ):
        while f >= 10:
            f = f/10.
            n += 1
    elif ( f < 1 ):
        while f <= 1:
            f = f*10.
            n -= 1
    return n



################################################################################
#~~~~~~~~~Legend generator
################################################################################
def addLegendAnnotation( annotation, text_pos=(0,0), arrow_pos=(0,0) ):
    ax = plt.gca()

    ax.annotate( annotation, xy=arrow_pos, xytext=text_pos,
                arrowprops=dict(facecolor='black', arrowstyle='->'))


################################################################################
#~~~~~~~~~Object plotter
################################################################################
def plotObj2D( objDict, newFigure=True, figArgs={} ):
    objs = objDict.keys()
    for idx,obj in enumerate(objs):
        if idx == 0:
            linePlot( newFigure=newFigure,
                     **merge_dictionaries( figArgs, objDict[obj] ) )
        else:
            linePlot( newFigure=False,
                     **merge_dictionaries( figArgs, objDict[obj] ) )


################################################################################
#~~~~~~~~~2D Vector plotting
################################################################################

def triangleMarker( phi_degrees = 0 ):
    return (3,0,phi_degrees-90)

def addLineOrientation( aline, at_point='end', rev_orientation=False, color='k', **kwargs ):
    try:
        aline['yData']
    except KeyError:
        aline = aline[ aline.keys()[0] ]

    if at_point == 'end':
        idx=aline['yData'].shape[0]-1
        theta = np.arctan2( aline['yData'][idx]-aline['yData'][idx-1] , ( aline['xData'][idx]-aline['xData'][idx-1] ) )
    elif at_point == 'front':
        idx=0
        theta = np.arctan2( aline['yData'][1]-aline['yData'][0] , ( aline['xData'][1]-aline['xData'][0] ) )
    elif at_point == 'middle':
        N = aline['yData'].shape[0]
        idx = int(N/2)
        theta = np.arctan2( aline['yData'][idx]-aline['yData'][idx-1] , ( aline['xData'][idx]-aline['xData'][idx-1] ) )

    if rev_orientation:
        theta = theta+np.pi

    plt.scatter( [aline['xData'][idx]],[aline['yData'][idx]],marker=triangleMarker( theta*180./np.pi ),c=color )

def addLabel( aline, label='x', at_point='end', radius=None, buffer_angle=0, color='k',**kwargs ):
    try:
        aline['yData']
    except KeyError:
        aline = aline[ aline.keys()[0] ]

    if at_point == 'end':
        idx=aline['yData'].shape[0]-1
        theta = np.arctan2( aline['yData'][idx]-aline['yData'][idx-1] , ( aline['xData'][idx]-aline['xData'][idx-1] ) )
    elif at_point == 'front':
        idx=0
        theta = np.arctan2( aline['yData'][1]-aline['yData'][0] , ( aline['xData'][1]-aline['xData'][0] ) )
    elif at_point == 'middle':
        N = aline['yData'].shape[0]
        idx = int(N/2)
        theta = np.arctan2( aline['yData'][idx]-aline['yData'][idx-1] , ( aline['xData'][idx]-aline['xData'][idx-1] ) )

    theta = theta + np.pi/2.+buffer_angle*np.pi/180.
    if radius is None:
        radius = 0.2*np.linalg.norm( [aline['xData'][idx] , aline['yData'][idx]] )

    xt = aline['xData'][idx] + radius*np.cos(theta)
    yt = aline['yData'][idx] + radius*np.sin(theta)

    plt.text( x=xt,y=yt,s=label, color=color )

def plot3DAxes( labels=['x','y','z'], altLabels=['','',''] ):

    xax = generateLine3D( x0=0, x1=1, y0=0., y1=0, z0=0, z1=0 )
    yax = generateLine3D( x0=0, x1=0, y0=0, y1=1, z0=0, z1=0 )
    zax = rotateObj3D(  generateLine3D( x0=0, x1=0, y0=0, y1=0, z0=0, z1=1, N=3 ), gamma=0, theta=90., phi=225. )

    figArgs = { 'mute_kwargs':True, 'xLims':[-1,1], 'yLims':[-1,1], 'nxTicks':5, 'nyTicks':5, 'showAxes':False, 'squareAxes':True }

    plotObj2D( xax, figArgs=figArgs )
    plt.scatter( [1],[0], marker=triangleMarker(0) )
    # plt.text( x=.9,y=.1,s=r'$\vec{v}$' )
    addLabel( xax['line'], label=labels[0], radius=.1 )
    addLabel( xax['line'], label=altLabels[0], radius=None, buffer_angle=180 )

    plotObj2D( yax, newFigure=False, figArgs=figArgs )
    plt.scatter( [0],[1], marker=triangleMarker(90) )
    addLabel( yax['line'], label=labels[1], buffer_angle=0 )
    addLabel( yax['line'], label=altLabels[1], buffer_angle=180, radius=.1, at_point='end' )


    plotObj2D( zax , newFigure=False, figArgs=figArgs )
    # plt.scatter( [np.cos( 225.*np.pi/180. )],[np.sin( 225.*np.pi/180. )], marker=triangleMarker(225) )
    addLineOrientation( zax['line'], at_point='end', rev_orientation=False  )
    addLabel( zax['line'], label=labels[2], buffer_angle=180, radius=.15, at_point='end' )
    addLabel( zax['line'], label=altLabels[2], buffer_angle=0, radius=None, at_point='end' )

def fancyLine( x0=0,x1=0,y0=0,y1=0,z0=0,z1=0, N=3,
              newFigure=True, figArgs={}, color='k', linestyle='-', zorder=0,
              label='x', at_point='end',
              buffer_angle=0, radius=.2,
              arrowOn = True):

    rline = generateLine3D( x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, N=3 )

    plotObj2D(rline, newFigure = newFigure,
              figArgs=merge_dictionaries(figArgs, {'plotOptions':{'color':color,'linestyle':linestyle,'zorder':zorder}}) )

    addLabel( rline['line'], label=label, buffer_angle=buffer_angle, radius=radius, at_point=at_point, color=color )

    if arrowOn:
        addLineOrientation( rline['line'], at_point=at_point, rev_orientation=False, color=color  )

################################################################################
#~~~~~~~~~Colorbar
################################################################################

def getLastColorbar():
    ax=plt.gca()        #plt.gca() for current axis, otherwise set appropriately.
    im=ax.images        #this is a list of all images that have been plotted
    cb=im[-1].colorbar
    return cb

def removeLastColorbar():
    cb = getLastColorbar()
    cb.remove()
