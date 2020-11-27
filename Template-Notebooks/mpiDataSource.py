from psana import *
import time

expName = 'cxix40218'
runNum = 50
detName = 'jungfrau4M'
dsource = MPIDataSource('exp='+expName+':run='+str(runNum)+':smd')
det = Detector(detName)

smldata = dsource.small_data('out.h5',gather_interval=100)

partial_run_sum = None
for nevt,evt in enumerate(dsource.events()):
    t0 = time.time()
    raw = det.raw(evt)
    t1 = time.time()
    calib = det.calib(evt, cmpars=(7,0,0))
    t2 = time.time()
    print "raw, calib: ", nevt, t1-t0, t2-t1
    if calib is None: continue
    cspad_sum = calib.sum()      # number
    cspad_roi = calib[0][0][3:5] # array
    if partial_run_sum is None:
        partial_run_sum = cspad_roi
    else:
        partial_run_sum += cspad_roi

    # save per-event data
    smldata.event(cspad_sum=cspad_sum,cspad_roi=cspad_roi)
    if nevt>120: break

# get "summary" data
run_sum = smldata.sum(partial_run_sum)
# save HDF5 file, including summary data
smldata.save(run_sum=run_sum)
