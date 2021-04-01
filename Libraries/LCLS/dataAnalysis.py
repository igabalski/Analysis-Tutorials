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
import time


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
        det =  Detector('CXI:LAS:MMN:04.RBV')
    try:
#         return det.values(evt)[0]
        return det(evt)
    except Exception as err:
        print(str(err))
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

    
def getAcqirisSum0( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns the summed acqiris signal in channel 1
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Acqiris sum
    '''
    if det is None:
        det = Detector('Acqiris') 
 
    try:
        return np.sum(det.waveform(evt)[0,:])
    except Exception:
        return None
    
    
def getAcqirisSum1( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns the summed acqiris signal in channel 1
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Acqiris sum
    '''
    if det is None:
        det = Detector('Acqiris') 
 
    try:
        return np.sum(det.waveform(evt)[1,:200])
    except Exception:
        return None

def getAcqirisSum2( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns the summed acqiris signal in channel 2
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Acqiris sum
    '''
    if det is None:
        det = Detector('Acqiris') 
 
    try:
        return np.sum(det.waveform(evt)[2,:])
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
    
def getWave8( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event (FWHM).
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event (FWHM).
    '''
    if det is None:
        det = Detector('CXI:DG2:BMMON:SUM')
    
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

def getDiodeTotalIntensity( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the value of the EPICS variable for the current event
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        The value of the EPICS variable for the current event (FWHM).
    '''
    if det is None:
        det = Detector('CXI-DG3-BMMON')
    
    try:
        return det.get(evt).TotalIntensity()
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
        det = Detector('CxiDg3_Ipm')
    
    try:
        return det.sum(evt)
    except Exception:
        return None

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
        # This is the energy after attenuation
        # Type help(det.get(evt)) in detectors.ipynb to see other options
        return det.get(evt).f_11_ENRC()
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

    
def getGasPressure( evt, det, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the defaultGet.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        defaultGet object (unknown type)
    '''
    
    try:
        return det(evt)
    except Exception:
        return None
    
def getXrayOn( evt, det, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the defaultGet.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        defaultGet object (unknown type)
    '''
    
    try:
        evrs = det(evt)
        return (162 not in evrs)
    except Exception:
        return None
    
def getLaserOn( evt, det, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None  ):
    '''
    Description: This function takes detector and event. Returns the defaultGet.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        defaultGet object (unknown type)
    '''
    
    try:
        evrs = det(evt)
        return 183 in evrs
    except Exception:
        return None
    
    
# @memorizeGet
def getCSPAD( evt, det = None, run=74, experiment='xppl2816', seconds=None,
             nanoseconds=None, fiducials=None, detType='CSPAD',
            threshold=200):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    try:
        data = det.calib(evt, mbits=39, cmpars=(7,0,0))
        data[(data < threshold) | (data>2000)] = 0
        return data
    except Exception as e:
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
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    try:
        return np.nansum(det.calib(evt, cmpars=(7,0,0)).flatten())
    except Exception:
        return None

def getCSPADrois( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    try:
        det_image = det.calib(evt, cmpars=(7,0,0))
        roi = det_image[0,206:306,462:572]
        roi0 = np.nansum(roi.flatten())
        roi = det_image[1,206:306,462:572]
        roi1 = np.nansum(roi.flatten())
        roi = det_image[2,206:306,462:572]
        roi2 = np.nansum(roi.flatten())
        roi = det_image[3,206:306,462:572]
        roi3 = np.nansum(roi.flatten())
        roi = det_image[4,206:306,462:572]
        roi4 = np.nansum(roi.flatten())
        roi = det_image[5,206:306,462:572]
        roi5 = np.nansum(roi.flatten())
        roi = det_image[6,206:306,462:572]
        roi6 = np.nansum(roi.flatten())
        roi = det_image[7,206:306,462:572]
        roi7 = np.nansum(roi.flatten())
        return [roi0,roi1,roi2,roi3,roi4,roi5,roi6,roi7]
    except Exception:
        return None
    
    
def getRadialrois( evt, det = None, run=74, threshold=100, corrections=None, binning_indices=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    if corrections is None:
        corrections = np.ones((8,512,1024))
    if binning_indices is None:
        binning_indices = load_obj('/cds/home/i/igabalsk/TRXS-Run18/Libraries/radial_binning_indices')
    
    
    Qs, phis = binning_indices.Qs, binning_indices.phis
    dQ, dphi = binning_indices.dQ, binning_indices.dphi
    roi_indices = binning_indices.roi_indices
    
    rois = np.zeros((len(Qs)))
    t0 = time.time()
    image = det.calib(evt, cmpars=(7,0,0),mbits=39)
    if image is None:
        return rois
    t1 = time.time()
#     print 'grab time: ',t1-t0
    
    # Correct for Thomson and Geometry
    image[(image < threshold) | (image>1000)] = 0.
    image = image*corrections
    for qidx, q in enumerate(Qs):
        roi = image[roi_indices[qidx]]
        try:
            rois[qidx]=np.nanmean( roi )
        except Exception as e:
            print(e)
            rois[qidx]=0.
    t2 = time.time()
#     print 'bin time: ',t2-t1
#     print 'roi mean:',np.nanmean(rois)
    return rois

def getQbinnedrois( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    QQ = np.load('/cds/home/i/igabalsk/TRXS-Run18/Libraries/QQ.npy')
    Qmin, Qmax = 0.5, 4.2
    NQ = 100
    dQ = float(Qmax-Qmin)/NQ
    Qs = np.linspace(Qmin,Qmax,NQ)
    rois = np.zeros(Qs.shape)
    t0 = time.time()
    image = det.calib(evt, cmpars=(7,0,0),mbits=39)
    if image is None:
        return rois
    t1 = time.time()
#     print 'read time: ', t1-t0
    t2 = time.time()
    for idx, q in enumerate(Qs):
        roi = image[ (QQ>q)&(QQ<q+dQ ) ].flatten()
#         print(idx,roi.size)
            
        try:
#             rois[idx] = np.nanmax( roi )
            roi[(roi < 75) | (roi>2000)] = 0
            rois[idx]=np.nanmean( roi )
        except Exception as e:
            print(e)
            rois[idx]=0.0
    t3 = time.time()
#     print 'mask time: ',t3-t2
    rois[rois==np.nan]=0
    return rois

def getQPhirois( evt, det = None, run=74, threshold=200, corrections=None, binning_indices=None):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    if corrections is None:
        corrections = np.ones((8,512,1024))
    if binning_indices is None:
        binning_indices = load_obj('/cds/home/i/igabalsk/TRXS-Run18/Libraries/binning_indices')
    
    
    Qs, phis = binning_indices.Qs, binning_indices.phis
    dQ, dphi = binning_indices.dQ, binning_indices.dphi
    roi_indices = binning_indices.roi_indices
    
    rois = np.zeros((len(Qs),len(phis)))
    image = det.calib(evt, cmpars=(7,0,0),mbits=39)
    if image is None:
        return rois
    # Correct for Thomson and Geometry
    image[(image < threshold) | (image>2000)] = 0.
    image[(image >= threshold) & (image<=2000)] = 1.
    image = image*corrections
    t4 = time.time()
    for qidx, q in enumerate(Qs):
        for phiidx, phi in enumerate(phis):
            roi = image[roi_indices[qidx][phiidx]]
            try:
                rois[qidx,phiidx]=np.nanmean( roi )
            except Exception as e:
                print(e)
                rois[qidx,phiidx]=0.
    return rois

def getPhotonHistogram( evt, det = None, run=74, experiment='xppl2816', seconds=None, nanoseconds=None, fiducials=None ):
    '''
    Description: This function takes detector and event. Returns per-pixel array of calibrated data intensities.
    
    Input:
        det: The psana detector object
        evt: psana event object
        
    Output:
        Per-pixel array of calibrated data intensities.
    '''
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')

    image = det.calib(evt,  mbits=39)
    roi = image[ ~np.isnan(image) ].flatten()
    hist,edges = np.histogram(roi, bins=200, range=(0,2000))
    return hist

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
    if det is not None:
        pass
    elif detType == 'CSPAD':
        det = Detector('cspad')
    elif detType == 'Jungfrau' and det is None:
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    try:
        return np.nanmedian(det.calib(evt, cmpars=(7,0,0)).flatten())
    except Exception:
        return None

# @memorizeGet
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
    elif det is 'CSPAD':
        det = Detector('cspad')
    elif det is 'Jungfrau':
        det = Detector('jungfrau4M')
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
#     detArrays = { name:np.zeros((eventMax,1)) for name in detList }
    detArrays = { name:[0 for idx in range(eventMax)] for name in detList }
    
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
        
def sumCSPAD( cspad, cspadMask=None, detType ='CSPAD' ):
    if cspadMask is None:
        cspadMask = createMask(detType=detType).astype(bool)
        
    theSum = 0 
    if detType == 'CSPAD':
        NTILE = 32
    elif detType == 'Jungfrau':
        NTILE = 8
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    for iTile in range(NTILE):
        cspadTile = cspad[iTile,:,:]
        tileMask = cspadMask[iTile,:,:]
        theSum += np.sum(cspadTile[tileMask].flatten())      
        
    return theSum

def CSPADgeometry( experiment='xppl2816' , run=72, detType = 'CSPAD' ):
    """
    Outputs x,y pixel coords of CSPAD
    """
    ds = DataSource('exp=%s:run=%d:smd' % (experiment, run))
    evt0 = ds.events().next()
    return getCSPADcoords(evt0, det = detType,  experiment=experiment , run=run)

#################################################################################################
# CSPAD mask generation
#################################################################################################
        
def createMask( experiment='xppl2816' , run=72, detType ='CSPAD' ):
    """
    Generates a mask of the CSPAD using a combination of the bad tiles, edges, and unbonded pixels.
    Also includes neighbors of the above.
    """
    print(experiment,run)
    ds = DataSource('exp=%s:run=%d:smd' % (experiment, run))
    evt0 = ds.events().next()
    if detType == 'CSPAD':
        detName = 'cspad'
    elif detType == 'Jungfrau':
        detName = 'jungfrau4M'
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
        
    CSPADStream=Detector( detName )
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

