from IPython.display import Audio
from IPython.display import clear_output

import os
import shlex, subprocess
import threading
import time
import re
import pickle

#################################################################################################
# Check for install directory
#################################################################################################
try:
    os.environ['INSTALLPATH']
except KeyError as e:
    print('Did you remember to specify os.environ[\'INSTALLPATH\']?')
    raise e



#################################################################################################
# Batch job terminal commands
#################################################################################################

def unixCMD( cmd ):
    '''
    Description: Submits a unix command through python

    Inputs:
        cmd: A unix command formatted as a string

    Outputs:
        stdoutdata: The output of the command
        stderrdata: Any resulting error. If no error, returns empty string.\
    '''
    cmd='ssh psana \'%s; exit\'' % cmd
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    (stdoutdata, stderrdata) = process.communicate()
    return stdoutdata, stderrdata

def bjobs():
    '''
    Description: Checks the status of the batch queue

    Inputs:
        None

    Outputs:
        stdoutdata: Current batch jobs running
        stderrdata: Any resulting error. If no error, returns empty string.\
    '''
    cmd='ssh psana \'bjobs; exit\''
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    (stdoutdata, stderrdata) = process.communicate()
    return stdoutdata, stderrdata

def bkill( jobid = None , killAll = False ):
    '''
    Description: Kills a specific batch job or all batch jobs

    Inputs:
        jobid: The batch jobid formatted as a string
        killAll: (boolean) True / False

    Outputs:
        stdoutdata: The result of bkill
        stderrdata: Any resulting error. If no error, returns empty string.\
    '''
    cmd = ''
    if jobid is not None:
        cmd='ssh psana \'bkill %s; exit\'' % jobid
    if killAll:
        cmd='ssh psana \'bkill 0; exit\''

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    (stdoutdata, stderrdata) = process.communicate()
    return stdoutdata, stderrdata

def checkActive(jobid):
    '''
    Description: Checks if a given jobid is still running in the batch queue

    Inputs:
        jobid (string): batch jobid

    Outputs:
        True / False
    '''
    stdoutdata, stderrdata = bjobs()
    return ( jobid in stdoutdata )

def extractJobId(batchOutput):
    '''
    Description: Extracts a jobid from the result of SubmitBatchJob

    Inputs:
        batchOutput (string): Output from SubmitBatchJob

    Outputs:
        jobid (string)
    '''
    m = re.search('<\d*>', batchOutput)
    return m.group(0)[1:-1]

def checkjobstatus():
    '''
    Description: This function generates a table from which information about the batch submission can be obtained.

    Inputs:
        None

    Outputs:
        A table with headers: JobID, User, Queue, From_Host, Exec_Host, Job_Name, Submit_Time
        A word to describe the current state of the batch submission: 'Running', 'Gathering', 'Remembered', 'Stopped'
    '''
    stdoutdata, stderrdata = bjobs()
    print stdoutdata
    print stderrdata
    print batchThreads.status
    
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
# Submit batch job function
#################################################################################################

def SubmitBatchJob(Job,RunType='python2',Nodes=32,Memory=7000,Queue='psnehprioq',OutputName='temp'):
    '''
    Description: Submits a batch job from within python

    Inputs:
        Job (list of strings): List of strings specifying the code to be run in the batch.
            Will be used to generate execution code.
        RunType (string): Execution type, eg. matlab / mpirun python2 / ./run.o
        Nodes (integer): Number of nodes required for batch job
        Memory (integer in MB): Amount of ram required on batch node
        Queue (string): Batch queue to submit job to
        OutputName (string): Specifies name of output file saved to os.environ['OUTPUTPATH']+USERNAME+'/Batch/Output'

    Outputs:
        jobid (string) of submitted job
    '''

    # Specify output directory
    OutputTo= BATCHDIR + '/Output/'+OutputName+'.out'
    BatchFileName= BATCHDIR +'/Python/'+'%s.py'%OutputName

    print "Deleting the old output file ..."
    process = subprocess.Popen(shlex.split("rm %s" % OutputTo), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print "Output: " + str(output)
    print "Error: " + str(error)

    print "Deleting the old executable file ..."
    process = subprocess.Popen(shlex.split("rm %s" % BatchFileName), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print "Output: " + str(output)
    print "Error: " + str(error)


    # Generate executable python file
    bfile = open(BatchFileName, 'w')
    for line in Job:
        bfile.write("%s \n" % line)
    bfile.close()
    
    # Execute batch command
    BatchCommand="ssh psana \'bsub -n %d -R \"rusage[mem=%d]\" -q %s -o %s %s/Libraries/pythonBatchMagic/BatchWrapper.sh %s %s; exit\'" % \
                                        (Nodes,Memory,Queue,OutputTo,os.environ['INSTALLPATH'],RunType,BatchFileName)


    print "Submitting: "+BatchCommand
    process = subprocess.Popen(BatchCommand, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    output, error = process.communicate()

#     nerror = 0
#     errorMax = 10
#     while ("Warning: job being submitted without an AFS token." not in error):
#         if len(error.strip()) == 0:
#             break
#         time.sleep(5)
#         print "Submitting again: "+BatchCommand
#         process = subprocess.Popen(BatchCommand, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
#         output, error = process.communicate()
#         nerror += 1
#         if nerror > errorMax:
#             print("Too many rejections")
#             break


    print "Output: " + str(output)
    print "Error: " + str(error)

    return extractJobId( str(output) )


#################################################################################################
# Threading for batch jobs
#################################################################################################

class batchThread (threading.Thread):
    '''
    Description: submits a batch job as a part of a thread
    After creating a batchThread, you can start it with batchThread.start()
    It can be stopped by batchThread.requestStop()

    Inputs:
        Job (list of strings): Defines job to be submitted

    Modifiable public variables:
        RunType (string): Execution type, eg. matlab / mpirun python2 / ./run.o
            Default: 'mpirun python2'
        Nodes (integer): Number of nodes required for batch job
            Default: 'mpirun python2'
        Memory (integer in MB): Amount of ram required on batch node
            Default: 'mpirun python2'
        Queue (string): Batch queue to submit job to
            Default: 'mpirun python2'
        OutputName (string): Specifies name of output file saved to os.environ['OUTPUTPATH']+USERNAME+'/Batch/Output'
            Default: 'temp'

    Other public variables:
        Status (string): 'Initialized', 'Running', 'Stopped', 'Finished'

    Outputs variables:
        None
    '''
    def __init__(self, Job):
        threading.Thread.__init__(self)

        # Specify job and batch job parameters
        self.Job = Job
#         self.RunType = 'mpirun python2'
        self.RunType = 'python2'
        self.Nodes = 1
        self.Memory = 7000
        self.Queue = 'psnehq'
        self.OutputName = 'temp'

        # Save internally the batch job id and run status
        self.jobid = None
        self.status = 'Initialized'
        self.flag = None


    def run(self):
        self.status = 'Running'
        
        self.jobid = SubmitBatchJob(self.Job ,
                                   RunType=self.RunType,
                                   Nodes=self.Nodes,
                                   Memory=self.Memory,
                                   Queue=self.Queue,
                                   OutputName=self.OutputName)

        forcedStop = False
        time.sleep(10)

        while checkActive( self.jobid ):
            if self.flag == 'Stop requested':
                bkill( jobid = self.jobid )
                forcedStop = True
                break

        if forcedStop:
            self.status = 'Stopped'
        else:
            self.status = 'Finished'


    def requestStop(self):
        self.flag = 'Stop requested'
