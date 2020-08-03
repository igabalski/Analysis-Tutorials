import numpy as np
from scipy import optimize

class chiSquareMinimization(object):
    def __init__(self, meanSignal,
                 stdSignal, model=None, reconType='CS'):
        """
        Initializes the reconstructor.

        Args:
            model: N x M matrix defining the Chi-Square problem
            meanSignal: vector of length M contianing mean-value of signal
            stdSignal: vector of length M containing standard deviaiton of signal
            reconType: 'CS' or 'B-CS'* for Chi-Square or bounded Chi-Square

            *'B-CS' requires additional configuration self.setBounds(bounds)

        Returns:
            The initialized reconstructor object
        """

        # Initialize the reconstruction parameters
        self.mu    = meanSignal
        self.sigma = stdSignal
        self.model     = model
        self.reconType = reconType

        # Optional parameters
        self.bounds = None

        # Initialize the output parameters
        self.solution     = {'mu':None, 'sigma':None, 'chi2':None}

    def updateInitialConditions(self,mu,sigma):
        self.mu = mu
        self.sigma=sigma

    def setBounds(self,bounds):
        if bounds.shape[0] != self.model.shape[0]:
            raise IndexError('bounds.shape[0] must match model.shape[0]')
        self.bounds = bounds

    def forceBounds(self,x):
        inRange = lambda x, b: (x > b[0])&(x < b[1])
        for idx, (xel,b) in enumerate(zip(x,self.bounds)):
            x[idx] = inRange(xel,b)*xel
        return x

    def generate_model(self):
        TT, WW = np.meshgrid(self.Ts, self.Ws)
        return np.cos(TT*WW)

    def reconstruct(self, x0=None):
        if (x0 is None) | (self.reconType =='CS'):
            x = self.solver('CS')
            if self.reconType == 'CS':
                self.update_solution(x)
                return None
            else:
                x0 = self.forceBounds(x['mu'])
        self.update_solution(self.solver(self.reconType, x0))

    def update_solution(self, x):
        self.solution['chi2']  = chiSquare(self.mu,self.sigma,self.model,x['mu'])
        self.solution['mu']    = x['mu']
        self.solution['sigma'] = x['sigma']

    def solver(self, solverType, x0=None):
        if solverType == 'CS':
            solution=chiSquareMin(self.mu,self.sigma,self.model)
            return {'mu':solution['xmin'],
                    'sigma':solution['mstd']}

        elif solverType == 'B-CS':
            A  = np.dot(self.model, np.diag(1/self.sigma))
            bl = map(lambda b: b[0]+b[1], self.bounds)
            Aw = np.dot(np.diag(bl),A).T
            res = optimize.lsq_linear(Aw, self.mu/self.sigma,
                             bounds=(0,1), lsq_solver='exact',lsmr_tol='auto', verbose=1,
                             max_iter=10000)
            return {'mu':bl*res.x,'sigma':None}





def chiSquare(mean,mean_std,model,x):
    """
    Calculates SUM(1/2 ||(Ax-b)/sigma||**2) where A=model, x=mean, sigma=mean_std.

    Args:
        mean: mean values of the measurement
        mean_std: standard deviation of the measurement at each index
        model: matrix describing your model for the measurement
        x: predicted vector

    Returns:
        cs
    """
    v0 = np.dot(x,model)
    vd = (v0 - mean) / mean_std
    return np.dot(vd,vd)



def chiSquareMin(mean,mean_std,model):
    """
    Calculates the global minimum for the unbounded Chi-squared problem described by
    min 1/2 ||(Ax-b)/sigma||**2 where A=model, x=mean, sigma=mean_std.

    Args:
        mean: mean values of the measurement
        mean_std: standard deviation of the measurement at each index
        model: matrix describing your model for the measurement

    Returns:
        {'xmin':xmin,'mstd':mstd, 'error_ratio':mstd/xmin,'cs':cs}
    """
    fx = np.dot(model, mean/mean_std**2)
    B  = np.dot(np.dot(model, np.diag(mean_std**-2)),model.transpose())

    try:
        Bi = np.linalg.solve(B.T.dot(B), B.T)
    except Exception:
        Bi = np.linalg.pinv(B)

    A  = np.dot(Bi,model)

    xmin = np.dot(A,mean/mean_std**2)

    mstd = np.dot(np.abs(A),mean_std)

    cs   = chiSquare(mean,mean_std,model,xmin)
    return {'xmin':xmin,'mstd':mstd, 'error_ratio':mstd/xmin,'cs':cs}
