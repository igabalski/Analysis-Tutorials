import numpy as np
################################################################################
#~~~~~~~~~Tiling
################################################################################
def tileLike(v,A):
    axis = 1
    if v.shape[0] == A.shape[1]:
        axis = 0
    return np.tile(v,(A.shape[axis],1)).reshape(A.shape)

def mean_subtract(A,axis=0):
    return A - np.tile(np.mean(A,axis=axis),(A.shape[axis],1)).reshape(A.shape)

def t0_subtract(A,axis=0):
    xslice = 0
    yslice = 0

    if axis==0:
        yslice = range(A.shape[1])
    else:
        xslice = range(A.shape[0])

    return A - np.tile(A[xslice,yslice],(A.shape[axis],1)).reshape(A.shape)

def tileMultiply(v,A):
    return A*tileLike(v,A)

################################################################################
#~~~~~~~~~Array slicing
################################################################################
def lineout( S, Q, Q0, axis=None ):
    idx = np.abs( Q-Q0 ).argmin()
    if axis == None:
        if S.shape[0] == Q.shape[0]:
            axis = 0
        else:
            axis = 1

    if axis == 0:
        return S[idx,:]
    else:
        return S[:,idx]
