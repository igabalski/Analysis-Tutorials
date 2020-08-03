"""
frequency analysis by chi-square library for diffuse scattering measurements
Author: Matthew R. Ware (mrware91@gmail.com)
Description: Tools to convert noisy time-resolved signal into Frequency space
"""

import numpy as np
from scipy import optimize
from chiSquare import *

class frequencyChiSquare(chiSquareMinimization):
    def __init__(self, Ts, meanSignal,
                 stdSignal, reconType='CS',
                 wMax=None, trigType='cos'):
        """
        Initializes the reconstructor.

        Args:

        Returns:
            The initialized reconstructor object
        """

        # Initialize the reconstruction parameters
        self.trigType = trigType
        self.Ts    = Ts
        N  = Ts.shape[0]
        if np.mod(N,2) == 0:
            raise ValueError('Class is only optimized for odd-length arrays')

        WMAX = np.pi/(Ts[1]-Ts[0])
        DW   = 2*np.pi/(Ts.max()-Ts.min())

        if wMax is None:
            self.Ws    = np.arange(0,WMAX+DW,DW)
        else:
            self.Ws    = np.arange(0,wMax+DW,DW)

        self.model     = self.generate_model()

        # Run initializaiton from parent class
        super(frequencyChiSquare, self).__init__(meanSignal,
                                                 stdSignal, self.model,
                                                 reconType)

    def setBounds(self,boundsFunction):
        self.bounds = boundsFunction(self.Ws)

    def generate_model(self):
        TT, WW = np.meshgrid(self.Ts, self.Ws)
        if self.trigType == 'cos':
            return np.cos(TT*WW)
        elif self.trigType == 'sin':
            return np.sin(TT*WW)
        elif self.trigType == 'both':
            return np.append(np.cos(TT*WW), np.sin(TT*WW), axis=0)
