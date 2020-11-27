import os
import shlex, subprocess
import threading
import time
import re
import pickle

try:
    os.environ['OUTPUTPATH']
except KeyError as e:
    print('Did you remember to specify os.environ[\'OUTPUTPATH\']?')
    raise e

try:
    os.environ['INSTALLPATH']
except KeyError as e:
    print('Did you remember to specify os.environ[\'INSTALLPATH\']?')
    raise e

# Load in the default LCLS libraries
import sys
sys.path.insert(0, os.environ['INSTALLPATH']+'/Libraries/LCLS')
from LCLSdefault import *

sys.path.insert(0, os.environ['INSTALLPATH']+'/Libraries/mattsTools')
from picklez import *

# Load in the get data library
from dataAnalysis import *

# Load in batch libraries
sys.path.insert(0, os.environ['INSTALLPATH']+'/Libraries/pythonBatchMagic')
from pythonBatchMagic import *

#################################################################################################
# Directory setup
#################################################################################################
try:
    os.environ['OUTPUTPATH']
except KeyError as e:
    print('Did you remember to specify os.environ[\'OUTPUTPATH\']?')
    raise e

currentUser, error = unixCMD("echo $USER")
currentUser = currentUser.strip()
print('Current user is '+currentUser+' will output batch to '+os.environ['OUTPUTPATH'] + '/%s/Batch' % currentUser)
BATCHDIR = os.environ['OUTPUTPATH'] + '/%s/Batch' % currentUser


# Make output directories if they do not exist
if not os.path.isdir(os.environ['OUTPUTPATH']+'/%s' % currentUser):
    os.mkdir(os.environ['OUTPUTPATH']+'/%s' % currentUser)

if not os.path.isdir(os.environ['OUTPUTPATH']+'/%s/Batch'% currentUser):
    os.mkdir(os.environ['OUTPUTPATH']+'/%s/Batch'% currentUser)

if not os.path.isdir(os.environ['OUTPUTPATH']+'/%s/Batch/Output'% currentUser):
    os.mkdir(os.environ['OUTPUTPATH']+'/%s/Batch/Output'% currentUser)

if not os.path.isdir(os.environ['OUTPUTPATH']+'/%s/Batch/Python'% currentUser):
    os.mkdir(os.environ['OUTPUTPATH']+'/%s/Batch/Python'% currentUser)

#################################################################################################
# Point data grabber
#################################################################################################

# code for individual nodes
def nodePointDataGrabber( eventMax=10, experiment='xppl2816', run=74, node=None, rank=None):
    '''
    Description: This function loads an input dictionary, dynamically generated
    by the point data batch thread class. It then pulls the point data from the
    run specified by the dictionary.

    Input:
        eventMax: Maximum number of events to read (integer)
            Default: 10
        experiment: Experiment name (string)
            Default: xppl2816
        run: Run number (integer)
            Default: 74
        node: Which node is the code running on
        rank: How many nodes is the batch job running across

    Output:
        None. Saves result to os.environ['OUTPUTPATH']+'/Batch/Output/nodePointDataGrabber%d-run-%d' % (node,run)

    '''

    detDict = load_obj( BATCHDIR +'/Output/detDict-run-%d'%run )

    print('Grabbing run %d from experiment %s') % (run , experiment)
    print('Node is %d, rank is %d' % (node, rank))

    # Create a data source to grab data from
    ds = DataSource('exp=%s:run=%d:smd' % (experiment , run) )

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
    detArrays = { name:np.zeros((int(np.ceil(float(eventMax)/float(rank))),1)) for name in detList }

    # for each detector (named), use .sum(evt) to grab data stored
    # store that in the dictionary
    nplaced =0
    for nevent, evt in enumerate(ds.events()):
        
        if nevent >= eventMax: break

        if np.mod( nevent,rank ) != node:
            continue

        # Always grab seconds, nanoseconds, fiducials to enable memorization
        seconds = getSeconds( evt )
        nanoseconds = getNanoseconds( evt )
        fiducials = getFiducials( evt )
        t0 = time.time()

        # Now grab user specified detectors
        for name in detList:
            getFunc = eval(detDict[name]['get-function'])
            detArrays[name][nplaced] =  getFunc( evt, detObjs[name],
                                                                     run=run, experiment=experiment,
                                                                     seconds=seconds, nanoseconds=nanoseconds, fiducials=fiducials)
        t1 = time.time()
        print 'Event: ',nplaced,', Time Elapsed: ',t1-t0
        nplaced = nplaced+1
        
    print('Done')
    
    save_obj( detArrays , BATCHDIR + '/Output/nodePointDataGrabber%d-run-%d' % (node,run) )


# thread for submitted
class batchPointDataGrabber (threading.Thread):
    '''
    Controls output to the nodePointDataGrabber nodes
    '''
    def __init__(self, detDict, eventMax=10, experiment='xppl2816', runNumber=74, rank=1 ):
        threading.Thread.__init__(self)

        # Specify function parameters
        self.detDict = detDict
        self.eventMax = eventMax
        self.experiment = experiment
        self.runNumber = runNumber
        self.batchRank = rank

        # Specify batch job parameters
        self.RunType = 'python2'
        self.Nodes = 1
        self.Memory = 1000
        self.Queue = 'psnehq'
        self.OutputName = 'pointData-run%d-temp' % runNumber

        # Save internally the batch job ids and run status
        self.subthreads = []
        self.status = 'Initialized'
        self.flag = None

        # Save the final output
        self.detArrays = None


    def run(self):
        self.status = 'Running'
        save_obj( self.detDict , BATCHDIR+'/Output/detDict-run-%d' % self.runNumber )
        time.sleep(1)

        for node in range(0,self.batchRank):
            try:
                os.remove( BATCHDIR+'/Output/nodePointDataGrabber%d-run-%d.pkl' % (node, self.runNumber) )
            except Exception:
                pass

        for node in range(self.batchRank):
            if self.flag == 'Stop requested':
                break

            batchJob = ['import os',
                        'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                        'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                        'import sys',
                        'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                        'from lclsBatch import *',
                        'nodePointDataGrabber( eventMax=%d, experiment=\'%s\', run=%d, node=%d, rank=%d)' % (self.eventMax, self.experiment,self.runNumber,node,self.batchRank)]

            myBatchThread = batchThread( batchJob )
            myBatchThread.RunType = self.RunType
            myBatchThread.Nodes = self.Nodes
            myBatchThread.Memory = self.Memory
            myBatchThread.Queue = self.Queue
            myBatchThread.OutputName = self.OutputName+'%d' % node
            myBatchThread.start()
            self.subthreads.append( myBatchThread )

            time.sleep(1)


        forcedStop = False
        time.sleep(10)

        while self.checkStatus():
            if self.flag == 'Stop requested':
                [athread.requestStop() for athread in self.subthreads]
                forcedStop = True
                break

        if forcedStop:
            self.status = 'Stopped'
        else:
            self.status = 'Gathering'
            self.gather()
            self.status = 'Finished'

    def resubmitMissing(self):
        for node in range(self.batchRank):
            try:
                load_obj( BATCHDIR + '/Output/nodePointDataGrabber%d-run-%d' % (node, self.runNumber) )
            except IOError as e:
                print('Resubmitting node %d' % node)
                self.resubmitNode( node )



    def resubmitNode(self, node):
        batchJob = ['import os',
                    'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                    'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                    'import sys',
                    'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                    'from lclsBatch import *',
                    'nodePointDataGrabber( eventMax=%d, experiment=\'%s\', run=%d, node=%d, rank=%d)' % (self.eventMax, self.experiment,self.runNumber,node,self.batchRank)]

        myBatchThread = batchThread( batchJob )
        myBatchThread.RunType = self.RunType
        myBatchThread.Nodes = self.Nodes
        myBatchThread.Memory = self.Memory
        myBatchThread.Queue = self.Queue
        myBatchThread.OutputName = self.OutputName+'%d' % node
        myBatchThread.start()
        self.subthreads.append( myBatchThread )

    def requestStop(self):
        self.flag = 'Stop requested'

    def checkStatus(self):
        if 'Running' in [athread.status for athread in self.subthreads]:
            return True
        elif 'Initialized' in [athread.status for athread in self.subthreads]:
            return True
        else:
            return False

    def gather(self):
        startNode = 0
        for node in range(0,self.batchRank):
            try:
                self.detArrays = load_obj( BATCHDIR + '/Output/nodePointDataGrabber%d-run-%d' % (node,self.runNumber) )
                startNode = node
                break
            except IOError:
                print('Could not read in node %d' % node)
        for node in range(startNode+1,self.batchRank):
            try:
                detArrays0 = load_obj( BATCHDIR + '/Output/nodePointDataGrabber%d-run-%d' % (node, self.runNumber) )
                for key in self.detArrays.keys():
                    self.detArrays[key] = np.append(self.detArrays[key], detArrays0[key])
#                     self.detArrays[key] += detArrays0[key]
            except IOError:
                print('Could not read in node %d' % node)




#################################################################################################
# CSPAD grabber (mean and variance done in different batch jobs)
#################################################################################################

def batchMeanCSPAD(node, experiment = 'xppl2816', runNumber = 72):
    tagDict = load_obj( BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,runNumber) )
    seconds, nanoseconds, fiducials = tagDict['seconds'], tagDict['nanoseconds'], tagDict['fiducials']

    ds = DataSource('exp=%s:run=%d:idx' % (experiment, runNumber))
    run = ds.runs().next()

    cspadMask = createMask( experiment=experiment , run=runNumber).astype(bool)

    integratedCSPAD = np.zeros((32,185,388))
    count = 0
    for sec,nsec,fid in zip(reversed(seconds.astype(int)),reversed(nanoseconds.astype(int)),reversed(fiducials.astype(int))):
        et = EventTime(int((sec<<32)|nsec),fid)
        evt = run.event(et)
        currCSPAD = getCSPAD(evt, run=runNumber, experiment=experiment,
                             seconds=sec, nanoseconds=nsec, fiducials=fid)
#         ipmIntensity = getIPM(evt,run=runNumber, experiment=experiment,
#                               seconds=sec, nanoseconds=nsec, fiducials=fid)
        ipmIntensity = sumCSPAD( currCSPAD , cspadMask )
        if currCSPAD is not None and ipmIntensity is not None:
                integratedCSPAD += currCSPAD / ipmIntensity
                count += 1

    cspadDict = { 'mean': integratedCSPAD/count , 'count': count }
    save_obj( cspadDict , BATCHDIR + '/Output/cspadDict-node-%d-run-%d' % (node,runNumber) )

def batchVarianceCSPAD(node, experiment = 'xppl2816', runNumber = 72):
    tagDict = load_obj( BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,runNumber) )
    seconds, nanoseconds, fiducials = tagDict['seconds'], tagDict['nanoseconds'], tagDict['fiducials']

    meanLoaded = False
    while not meanLoaded:
        time.sleep(1)
        try:
            mean = load_obj( BATCHDIR + '/Output/cspadDict-node-%d-run-%d' % (node, runNumber) )['mean']
            meanLoaded = True
        except Exception:
            pass

    ds = DataSource('exp=%s:run=%d:idx' % (experiment, runNumber))
    run = ds.runs().next()


    cspadMask = createMask( experiment=experiment , run=runNumber).astype(bool)

    varianceCSPAD = np.zeros((32,185,388))
    count = 0
    for sec,nsec,fid in zip(reversed(seconds.astype(int)),reversed(nanoseconds.astype(int)),reversed(fiducials.astype(int))):
        et = EventTime(int((sec<<32)|nsec),fid)
        evt = run.event(et)
        currCSPAD = getCSPAD(evt, run=runNumber, experiment=experiment,
                              seconds=sec, nanoseconds=nsec, fiducials=fid)
#         ipmIntensity = getIPM(evt, run=runNumber, experiment=experiment,
#                               seconds=sec, nanoseconds=nsec, fiducials=fid)
        ipmIntensity = sumCSPAD( currCSPAD , cspadMask )
        if currCSPAD is not None and ipmIntensity is not None:
            varianceCSPAD += (currCSPAD / ipmIntensity - mean)**2
            count += 1



    variance = varianceCSPAD/count
    save_obj( variance , BATCHDIR + '/Output/variance-node-%d-run-%d' % (node,runNumber) )


# thread for submitted
class batchCSPADGrabber (threading.Thread):
    def __init__(self, tagsList, experiment='xppl2816', runNumber=74):
        threading.Thread.__init__(self)

        # Specify function parameters
        self.tagsList = tagsList
        self.experiment = experiment
        self.runNumber = runNumber
        self.batchRank = len( tagsList )

        # Specify batch job parameters
        self.RunType = 'python2'
        self.Nodes = 1
        self.Memory = 7000
        self.Queue = 'psnehq'
        self.OutputName = 'cspadGrabber-run-%d-temp-' % runNumber

        # Save internally the batch job ids and run status
        self.subthreads = []
        self.status = 'Initialized'
        self.flag = None

        # Save the final output
        NT = len( tagsList )
        self.CSPAD = np.zeros((32,185,388,NT))
        self.variance = np.zeros((32,185,388,NT))
        self.counts = np.zeros((NT,1))


    def run(self):
        self.status = 'Running'

        for node in range(self.batchRank):
            try:
                os.remove( BATCHDIR + '/Output/variance-node-%d-run-%d.pkl' % node  % (node,self.runNumber) )
            except Exception:
                pass

            try:
                os.remove( BATCHDIR + '/Output/cspadDict-node-%d-run-%d.pkl' % (node,self.runNumber)  )
            except Exception:
                pass

        for node in range(self.batchRank):
            if self.flag == 'Stop requested':
                break

            save_obj( self.tagsList[node] , BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,self.runNumber) )

            # Submit CSPAD to batch
            batchJobC = ['import os',
                        'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                        'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                        'import sys',
                        'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                        'from lclsBatch import *',
                        'batchMeanCSPAD( node=%d, experiment=\'%s\', runNumber=%d )' % (node, self.experiment,self.runNumber)]
            self.submitBatch( batchJobC , self.OutputName+'CSPAD-%d' % node )

            time.sleep(5)

            # Submit variance of CSPAD to batch
            batchJobV = ['import os',
                        'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                        'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                        'import sys',
                        'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                        'from lclsBatch import *',
                        'batchVarianceCSPAD( node=%d, experiment=\'%s\', runNumber=%d )' % (node, self.experiment,self.runNumber)]
            self.submitBatch( batchJobV , self.OutputName+'variance-%d' % node )

            time.sleep(5)


        forcedStop = False
        time.sleep(10)

        while self.checkStatus():
            if self.flag == 'Stop requested':
                [athread.requestStop() for athread in self.subthreads]
                forcedStop = True
                break

        if forcedStop:
            self.status = 'Stopped'
        else:
            self.status = 'Gathering'
            self.gather()
            self.status = 'Finished'

    def submitBatch(self, job, outputName):
        myBatchThread = batchThread( job )
        myBatchThread.RunType = self.RunType
        myBatchThread.Nodes = self.Nodes
        myBatchThread.Memory = self.Memory
        myBatchThread.Queue = self.Queue
        myBatchThread.OutputName = outputName
        myBatchThread.start()
        self.subthreads.append( myBatchThread )

    def requestStop(self):
        self.flag = 'Stop requested'

    def checkStatus(self):
        if 'Running' in [athread.status for athread in self.subthreads]:
            return True
        elif 'Initialized' in [athread.status for athread in self.subthreads]:
            return True
        else:
            return False

    def gather(self):
        for node in range(self.batchRank):
            CSPADdict = load_obj( BATCHDIR + '/Output/cspadDict-node-%d-run-%d' % (node,self.runNumber) )
            self.CSPAD[:,:,:,node] = CSPADdict['mean']
            self.counts[node] = CSPADdict['count']
            self.variance[:,:,:,node] = load_obj( BATCHDIR + '/Output/variance-node-%d-run-%d' % (node,self.runNumber) )




#################################################################################################
# CSPAD grabber (mean and variance done in different batch jobs)
#################################################################################################

def batchMeanVarCSPAD(node, experiment = 'xppl2816', runNumber = 72, detType='CSPAD'):
    tagDict = load_obj( BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,runNumber) )
    seconds, nanoseconds, fiducials = tagDict['seconds'], tagDict['nanoseconds'], tagDict['fiducials']
    print seconds.shape, nanoseconds.shape, fiducials.shape
    
    ds = DataSource('exp=%s:run=%d:idx' % (experiment, runNumber))
    run = ds.runs().next()


    cspadMask = createMask( experiment=experiment , run=runNumber, detType=detType).astype(bool)
    
    if detType =='CSPAD':
        integratedCSPAD = np.zeros((32,185,388))
        varianceCSPAD = np.zeros((32,185,388))
        det = Detector('cspad')
    elif detType =='Jungfrau':
        integratedCSPAD = np.zeros((8,512,1024))
        varianceCSPAD = np.zeros((8,512,1024))
        det = Detector('jungfrau4M')
    else:
        raise ValueError('detType must be CSPAD or Jungfrau')
    
    count = 0
    print 'Mean and Variance (Welford algorithm)'
    for sec,nsec,fid in zip(reversed(seconds.astype(int)),reversed(nanoseconds.astype(int)),reversed(fiducials.astype(int))):
        et = EventTime(int((sec<<32)|nsec),fid)
        evt = run.event(et)
        t0 = time.time()
        currCSPAD = getCSPAD(evt, det=det, run=runNumber, experiment=experiment,
                             seconds=sec, nanoseconds=nsec, fiducials=fid, detType=detType)
        t1 = time.time()
        print 'Frame: ',count,', Grab Time: ', t1-t0
        ipmIntensity = sumCSPAD( currCSPAD , cspadMask, detType=detType )
        if currCSPAD is not None and ipmIntensity is not None:
            lastmeanCSPAD = integratedCSPAD/max(count,1)
            integratedCSPAD += currCSPAD #/ ipmIntensity
            thismeanCSPAD = integratedCSPAD/(count+1)
            varianceCSPAD = varianceCSPAD + (currCSPAD-lastmeanCSPAD)*(currCSPAD-thismeanCSPAD)
            count += 1

    mean = integratedCSPAD/count
    variance = varianceCSPAD/count
    
    cspadDict = { 'mean': integratedCSPAD/float(count) , 'count': count , 'variance': variance }

    save_obj( cspadDict , BATCHDIR + '/Output/mean-var-node-%d-run-%d' % (node,runNumber) )


# thread for submitted
class batchCSPADMVGrabber (threading.Thread):
    def __init__(self, tagsList, experiment='xppl2816', runNumber=74, detType='CSPAD'):
        threading.Thread.__init__(self)

        # Specify function parameters
        self.tagsList = tagsList
        self.experiment = experiment
        self.runNumber = runNumber
        self.batchRank = len( tagsList )

        # Specify batch job parameters
        self.RunType = 'python2'
        self.Nodes = 1
        self.Memory = 7000
        self.Queue = 'psnehq'
        self.OutputName = 'CSPADMV-run-%d-temp-' % runNumber

        # Save internally the batch job ids and run status
        self.subthreads = []
        self.status = 'Initialized'
        self.flag = None

        # Save the final output
        self.detType = detType
        NT = len( tagsList )
        if detType == 'CSPAD':
            self.CSPAD = np.zeros((32,185,388,NT))
            self.variance = np.zeros((32,185,388,NT))
        elif detType =='Jungfrau':
            self.CSPAD = np.zeros((8,512,1024,NT))
            self.variance = np.zeros((8,512,1024,NT))
        else:
            raise ValueError('detType must be CSPAD or Jungfrau')
        self.counts = np.zeros((NT,1))


    def run(self):
        self.status = 'Running'

        for node in range(self.batchRank):
            try:
                os.remove( BATCHDIR + '/Output/mean-var-node-%d-run-%d.pkl' % node  % (node,self.runNumber) )
            except Exception:
                pass

        for node in range(self.batchRank):
            if self.flag == 'Stop requested':
                break

            save_obj( self.tagsList[node] , BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,self.runNumber) )
            time.sleep(1)

            # Submit CSPAD to batch
            batchJobCV = ['import os',
                        'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                        'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                        'import sys',
                        'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                        'from lclsBatch import *',
                        'batchMeanVarCSPAD( node=%d, experiment=\'%s\', runNumber=%d,detType=\'%s\' )' % (node, self.experiment,self.runNumber,self.detType)]
            self.submitBatch( batchJobCV , self.OutputName+'CSPAD-%d' % node )

            time.sleep(1)


        forcedStop = False
        time.sleep(10)

        while self.checkStatus():
            if self.flag == 'Stop requested':
                [athread.requestStop() for athread in self.subthreads]
                forcedStop = True
                break

        if forcedStop:
            self.status = 'Stopped'
        else:
            self.status = 'Gathering'
            self.gather()
            self.status = 'Finished'

    def resubmitMissing(self):
        for node in range(self.batchRank):
            try:
                load_obj( BATCHDIR + '/Output/mean-var-node-%d-run-%d' % (node,self.runNumber) )
            except IOError as e:
                self.resubmitNode( node )


    def resubmitNode(self, node):
        save_obj( self.tagsList[node] , BATCHDIR + '/Output/tagDict-node-%d-run-%d' % (node,self.runNumber) )
        time.sleep(1)
        batchJobCV = ['import os',
                    'os.environ[\'INSTALLPATH\']=\'%s\''% os.environ['INSTALLPATH'],
                    'os.environ[\'OUTPUTPATH\']=\'%s\'' % os.environ['OUTPUTPATH'],
                    'import sys',
                    'sys.path.insert(0, os.environ[\'INSTALLPATH\']+\'/Libraries/pythonBatchMagic\')',
                    'from lclsBatch import *',
                    'batchMeanVarCSPAD( node=%d, experiment=\'%s\', runNumber=%d )' % (node, self.experiment, self.runNumber)]
        self.submitBatch( batchJobCV , self.OutputName+'CSPAD-%d' % node )

    def submitBatch(self, job, outputName):
        myBatchThread = batchThread( job )
        myBatchThread.RunType = self.RunType
        myBatchThread.Nodes = self.Nodes
        myBatchThread.Memory = self.Memory
        myBatchThread.Queue = self.Queue
        myBatchThread.OutputName = outputName
        myBatchThread.start()
        self.subthreads.append( myBatchThread )

    def requestStop(self):
        self.flag = 'Stop requested'

    def checkStatus(self):
        if 'Running' in [athread.status for athread in self.subthreads]:
            return True
        elif 'Initialized' in [athread.status for athread in self.subthreads]:
            return True
        else:
            return False

    def gather(self):
        for node in range(self.batchRank):
            try:
                CSPADdict = load_obj( BATCHDIR + '/Output/mean-var-node-%d-run-%d' % (node,self.runNumber) )
                self.CSPAD[:,:,:,node] = CSPADdict['mean']
                self.counts[node] = CSPADdict['count']
                self.variance[:,:,:,node] = CSPADdict['variance']
            except IOError as ioe:
                print(str(ioe))
                self.CSPAD[:,:,:,node] = np.nan
                self.counts[node] = np.nan
                self.variance[:,:,:,node] = np.nan
                
