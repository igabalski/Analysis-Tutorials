#################################################################################################

# dataAnalysis library by Matt Ware and Noor Al-Sayyad
# Last updated 10/16/18

# MIT License

# Copyright (c) 2018 Matthew Ware (mrware91@gmail.com) and Noor Al-Sayyad (nooral@stanford.edu)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#################################################################################################

# Load in required libaries
import numpy as np
from psana import *
import os, sys, pickle



#################################################################################################
# Loading and saving for memorization below
#################################################################################################

def load_obj(filename ):
    """
    Loads object from name.pkl and returns its value

    Args:
        filename: String designating directory and name of file, ie. /Folder/Filename, where Filename.pkl is the object

    Returns:
        The value of the object in filename
    """
    try:
        with open(filename + '.pkl', 'rb') as f:
            print filename+" remembered!"
            return pickle.load(f)
    except IOError as e:
        print "IOError: Did you load the correct file? %s" % filename
        raise e


def save_obj(obj, filename ):
    """
    Saves object from filename.pkl

    Args:
        obj: The python object to save
        filename: String designating directory and name of file, ie. /Folder/Filename, where Filename.pkl is the object
    """
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#################################################################################################
# Memorization to speed up analysis
#################################################################################################

def det_fn_name(fn ):
    idx1=str(fn).index('at')
    idx0=str(fn).index(' ')
    return str.strip(str(fn)[idx0:idx1])


def dict2List( aDict ):
    aList = []
    [ aList.extend([ key , aDict[key] ]) for key in aDict ]
    return aList

def dict2Tuple( aDict ):
    return tuple( dict2List(aDict) )

try:
    os.environ['OUTPUTPATH']
except KeyError as e:
    print('Did you remember to specify os.environ[\'OUTPUTPATH\']?')
    raise e

class memorizeGet:
    def __init__(self, fn):
        self.outputDir = os.environ['OUTPUTPATH']+'/Memories'
        
        self.fn = fn
        self.memo = {}
        self.fn_name=det_fn_name(fn)
        self.remember()
        
        
    def __call__(self, *args, **kwargs):
        newDict = kwargs.copy()
        try:
            newDict.pop('det')
        except KeyError as e:
            pass
        index = dict2List(newDict)
        index.sort()
        index = tuple(index)
        if index not in self.memo:
            self.memo[index] = self.fn(*args, **kwargs)
            self.make_memory()
        return self.memo[index]
    
    def make_memory(self):
        try:
            #currMemo = load_obj(self.outputDir +'/'+ self.fn_name)
            #self.memo.update(currMemo)
            save_obj(self.memo,self.outputDir +'/'+ self.fn_name)
        except IOError:
            os.mkdir(self.outputDir)
            save_obj(self.memo,self.outputDir +'/'+ self.fn_name)
        
    def remember(self):
        try:
            self.memo=load_obj(self.outputDir +'/'+ self.fn_name)
        except (IOError, EOFError) as e:
            pass
        
    def forget(self):
        self.memo={}
        os.remove( self.outputDir +'/'+ self.fn_name +'.pkl' )

#################################################################################################
# PSANA help function
#################################################################################################

def detInfo( detName, run=74, experiment='xppl2816' ):
    '''
    Description: This function takes detector name, run number, and experiment. Returns help(detector).
    
    Input:
        detName: detector name
        run: run number
        experiment: experiment name
        
    Output:
        help(detector)
    '''
    ds = DataSource('exp=%s:run=%d:smd' % (experiment , run) )
    det = Detector(detName)
    help(det)
    
def getEvt0Data( detName, detFunc = lambda x,evt: x.get(evt), run=74, experiment='xppl2816' ):
    ds = DataSource('exp=%s:run=%d:smd' % (experiment , run) )
    det = Detector(detName)
    evt0 = ds.events().next()
    
    return detFunc( det , evt0 )

#################################################################################################
# Detector GET functions
#################################################################################################

# @memorizeGet
def getStageEncoder( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns the calibrated encoder values in channel 0. 
                 The calibrated value is calculated for each channel as follows: value = scale * (raw_count + offset)
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Calibrated encoder values in channel 0
    '''
    if det is None:
        det =  Detector('XppEndstation.0:USDUSB.0')
    try:
        return det.values(evt)[0]
    except Exception:
        return None

# @memorizeGet
def getTTFltPos( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event.
    '''
    if det is None:
        det = Detector('XPP:TIMETOOL:FLTPOS')
        
    try:
        return det(evt)
    except Exception:
        return None

# @memorizeGet
def getEbeamCharge( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns the charges (nC).
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Charges (nC)
    '''
    if det is None:
        det = Detector('EBeam') 
 
    try:
        return det.get(evt).ebeamCharge()
    except Exception:
        return None

# @memorizeGet
def getFltPosFWHM( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event (FWHM).
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event (FWHM).
    '''
    if det is None:
        det = Detector('XPP:TIMETOOL:FLTPOSFWHM')
    
    try:
        return det(evt)
    except Exception:
        return None

# @memorizeGet
def getTTAMPL( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event (FWHM).
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event (FWHM).
    '''
    if det is None:
        det = Detector('XPP:TIMETOOL:AMPL')
    
    try:
        return det(evt)
    except Exception:
        return None

# @memorizeGet
def getTTREFAMPL( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event (FWHM).
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event (FWHM).
    '''
    if det is None:
        det = Detector('XPP:TIMETOOL:REFAMPL')
    
    try:
        return det(evt)
    except Exception:
        return None


# @memorizeGet
def getSeconds( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes event. Returns time (seconds).
    
    Input:
        evt: psana event object
        
    Output:
        Time (seconds)
    '''
    evtId = evt.get(EventId)
    return evtId.time()[0]

# @memorizeGet
def getNanoseconds( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes event. Returns time (nanoseconds).
    
    Input:
        evt: psana event object
        
    Output:
        Time (nanoseconds)
    '''
    evtId = evt.get(EventId)
    return evtId.time()[1]

# @memorizeGet
def getFiducials( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes event. Returns fiducials.
    
    Input:
        evt: psana event object
        
    Output:
        Fiducials
    '''
    evtId = evt.get(EventId)
    return evtId.fiducials()

# @memorizeGet
def getIPM( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Intensity (float)
    '''
    if det is None:
        det = Detector('XppSb3_Ipm')
    
    try:
        return det.sum(evt)
    except Exception:
        return None

# @memorizeGet
'''
old method, old detector ID
def getXrayEnergy( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    
    Description: This function takes detector and event. Returns the intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Intensity (float)
    
    if det is None:
        det = Detector('SIOC:SYS0:ML00:AO541')
    
    try:
        return det(evt)
    except Exception:
        return None
'''

# current detector ID
def getXrayEnergy( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Intensity (float)
    '''
    if det is None:
        det = Detector('FEEGasDetEnergy')
    
    try:
        return det(evt)
    except Exception:
        return None

# @memorizeGet
def getXPos( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns beam position along the x-axis.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The estimated x-position
    '''
    if det is None:
        det=Detector('XppSb3_Ipm')
    
    try:
        return det.xpos(evt)
    except Exception:
        return None
        

# @memorizeGet
def getYPos( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns beam position along the x-axis.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The estimated x-position
    '''
    if det is None:
        det=Detector('XppSb3_Ipm')
        
    try:
        return det.ypos(evt)
    except Exception:
        return None
    


# @memorizeGet
def getDefault( evt, det, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the defaultGet.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        defaultGet object (unknown type)
    '''
    
    try:
        return det.get(evt)
    except Exception:
        return None

# @memorizeGet
def getCSPAD( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is None:
        det = Detector('cspad')
    try:
        return det.calib(evt)
    except Exception:
        return None

# @memorizeGet
def getCSPADsum( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is None:
        det = Detector('cspad')
    try:
        return np.nansum(det.calib(evt).flatten())
    except Exception:
        return None

# @memorizeGet
def getCSPADmedian( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is None:
        det = Detector('cspad')
    try:
        return np.nanmedian(det.calib(evt).flatten())
    except Exception:
        return None

@memorizeGet
def getCSPADcoords( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns per-pixel arrays of x coordinates and y coordinates.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of x coordinates
        Per-pixel array of y coordinates
    '''
    if det is None:
        det = Detector('cspad')
    return det.coords_x(evt) , det.coords_y(evt)

#################################################################################################
# Collect point data from run
#################################################################################################

def pointDataGrabber( detDict, eventMax=10, experiment='xppl2816', run=74 ):
    '''
    Description: This function takes in a list of detector names, 
    specifies number of events, which experiment, and which run to look at
    
    Input:
        detList: A list of strings that give the detector names
        eventMax: Maximum number of events to read (integer)
            Default: 10
        experiment: Experiment name (string)
            Default: xppl2816
        run: Run number (integer)
            Default: 74
    
    Output:
        detArrays: Dictionary of arrays with grab data
    
    '''
    

    print('Grabbing run %d from experiment %s') % (run , experiment)
    
    # Create a data source to grab data from
    ds = DataSource('exp=%s:run=%d' % (experiment , run) )
    
    # Generate detList from detDict
    detList = detDict.keys() 
    
    # Create empty dictionary to store names of detectors in
    detObjs = {name:'' for name in detList} 
    
    for name in detList:
        # gets name of detectors and stores in dictionary
        try:
            detObjs[name] = Detector( detDict[name]['name'] )
        except KeyError as e:
            detObjs[name] = None
    
    
    # Create empty dictionary to store
    detArrays = { name:np.zeros((eventMax,1)) for name in detList }
    
    # for each detector (named), use .sum(evt) to grab data stored
    # store that in the dictionary
    for nevent, evt in enumerate(ds.events()):
        # Always grab seconds, nanoseconds, fiducials to enable memorization
        seconds = getSeconds( evt )
        nanoseconds = getNanoseconds( evt )
        fiducials = getFiducials( evt )
        
        # Now grab user specified detectors
        for name in detList:
            detArrays[name][nevent] =  detDict[name]['get-function']( evt, detObjs[name], 
                                                                     run=run, experiment=experiment, 
                                                                     seconds=seconds, nanoseconds=nanoseconds, fiducials=fiducials)
        if nevent == eventMax-1: break
        
    return detArrays
    
    
#################################################################################################
# Grab CSPAD data
#################################################################################################
    
def meanCSPAD(seconds, nanoseconds, fiducials, experiment = 'xppl2816', runNumber = 72):
    ds = DataSource('exp=%s:run=%d:idx' % (experiment, runNumber))
    run = ds.runs().next()
    integratedCSPAD = np.zeros((32,185,388))
    count = 0
    for sec,nsec,fid in zip(reversed(seconds.astype(int)),reversed(nanoseconds.astype(int)),reversed(fiducials.astype(int))):
        et = EventTime(int((sec<<32)|nsec),fid)
        evt = run.event(et)
        currCSPAD = getCSPAD(evt, run=runNumber, experiment=experiment,
                             seconds=sec, nanoseconds=nsec, fiducials=fid)
        ipmIntensity = getIPM(evt,run=runNumber, experiment=experiment,
                              seconds=sec, nanoseconds=nsec, fiducials=fid)
        if currCSPAD is not None and ipmIntensity is not None:
                integratedCSPAD += currCSPAD / ipmIntensity
                count += 1
    return integratedCSPAD/count, count

def varianceCSPAD(mean, seconds, nanoseconds, fiducials, experiment = 'xppl2816', runNumber = 72):
    ds = DataSource('exp=%s:run=%d:idx' % (experiment, runNumber))
    run = ds.runs().next()
    varianceCSPAD = np.zeros((32,185,388))
    count = 0
    for sec,nsec,fid in zip(reversed(seconds.astype(int)),reversed(nanoseconds.astype(int)),reversed(fiducials.astype(int))):
        et = EventTime(int((sec<<32)|nsec),fid)
        evt = run.event(et)
        currCSPAD = getCSPAD(evt, run=runNumber, experiment=experiment,
                              seconds=sec, nanoseconds=nsec, fiducials=fid)
        ipmIntensity = getIPM(evt, run=runNumber, experiment=experiment,
                              seconds=sec, nanoseconds=nsec, fiducials=fid)
        if currCSPAD is not None and ipmIntensity is not None:
            varianceCSPAD += (currCSPAD / ipmIntensity - mean)**2
            count += 1
    return varianceCSPAD/count 

#################################################################################################
# Filter generation
#################################################################################################
    
def mad( anNPArray ):
    '''
    Description: This function takes an array. Returns the median absolute deviation.
                 The median absolute deviation is the median of the deviations from the median.
    
    Input:
        anNPArray: an array
        
    Output:
        median absolute deviation (float)
    '''
    median = np.median( anNPArray )
    med_abs_dev = np.median(np.abs(anNPArray - median))
    
    return med_abs_dev

def ingroup( anNParray , maddevs = 3 ):
    '''
    Description: This function takes an array. Returns an array of 0s and 1s corresponding to bad and good values respectively in array.
                 Bad values are outside of lower (LB) and upper bounds (UB). Good values are inside LB and UB. 
    
    Input:
        anNPArray: an array
        
    Output:
        an array of 0s and 1s corresponding to bad and good values in array (boolean)
    '''
    anNParray[ np.isnan(anNParray) ] = 0
    UB = np.median(anNParray) + maddevs*mad(anNParray)
    LB = np.median(anNParray) - maddevs*mad(anNParray)
    return ((anNParray >= LB) & (anNParray <= UB) )

def runFilter( pointData, filterOn = ['xint3','ebeamcharge','fltposfwhm'], maddevs=3 ):
    runfilter=np.ones_like( pointData[filterOn[0]] )
    
    for var in filterOn:
        runfilter = runfilter * ingroup( pointData[var] , maddevs = maddevs )
        
    return runfilter
    
    
#################################################################################################
# CSPAD plotting
#################################################################################################
        
def sumCSPAD( cspad, cspadMask=None ):
    if cspadMask is None:
        cspadMask = createMask().astype(bool)
        
    theSum = 0 
    for iTile in range(32):
        cspadTile = cspad[iTile,:,:]
        tileMask = cspadMask[iTile,:,:]
        theSum += np.sum(cspadTile[tileMask].flatten())      
        
    return theSum

def CSPADgeometry( experiment='xppl2816' , run=72 ):
    """
    Outputs x,y pixel coords of CSPAD
    """
    ds = DataSource('exp=%s:run=%d:smd' % (experiment, run))
    evt0 = ds.events().next()
    return getCSPADcoords(evt0)

#################################################################################################
# CSPAD mask generation
#################################################################################################
        
def createMask( experiment='xppl2816' , run=72 ):
    """
    Generates a mask of the CSPAD using a combination of the bad tiles, edges, and unbonded pixels.
    Also includes neighbors of the above.
    """
    ds = DataSource('exp=%s:run=%d:smd' % (experiment, run))
    evt0 = ds.events().next()
    CSPADStream=Detector( 'cspad' )
    CSPAD_mask_geo=CSPADStream.mask_comb(evt0,mbits=37)
    CSPAD_mask_bad_pixels=np.multiply(CSPAD_mask_geo,CSPADStream.mask_geo(evt0,mbits=15))
    CSPAD_mask_edges=CSPADStream.mask_edges(CSPAD_mask_bad_pixels,mrows=4,mcols=4)
    return CSPAD_mask_edges



#################################################################################################
# ROI operations
#################################################################################################
def roiSummed( x0, y0, dx, dy, x, y, image ):
    idx = ( x0 < x ) & ( (x0+dx) > x ) & ( y0 < y ) & ( (y0+dy) > y )
    return np.sum( image[idx , :] , 0 )

#################################################################################################
# Sample use of batch job function
#################################################################################################

