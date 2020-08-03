"""
Converts input data from momentum to real space by fitting to a spherical
bessel model
Author: Matthew R. Ware (mrware91@gmail.com)
Description: Tools to convert noisy, undersampled Fourier-space distributions
into real-space images.
"""

import numpy as np
from scipy import optimize, special
from chiSquare import *

import sys
sys.path.insert(0, '../formFactors')
from formFactors import *

class sphericalBesselChiSquare(chiSquareMinimization):
    def __init__(self, Qs, Atoms, meanSignal,
                 stdSignal, legendreOrder=0,
                 moleculeType='homonuclear', reconType='CS',
                 rMax=10):
        """
        Initializes the reconstructor.

        Args:
            Qs: The grid of momentum transfer coordinates
            Atoms: The atoms in your molecule*
            meanSignal: The measured signal at the coordinates Qs
            stdSignal:  The standard deviation of that signal

        *You'll need to add the definition of f(Q) for your molecule to the library

        Returns:
            The initialized reconstructor object
        """
        # Initialize the reconstruction parameters
        self.Qs    = Qs
        N  = Qs.shape[0]
        RMAX = np.pi/(Qs[1]-Qs[0])
        DR   = 2*np.pi/(Qs.max()-Qs.min())
        if RMAX > rMax:
            self.Rs    = np.arange(DR,rMax+DR,DR)
        else:
            self.Rs    = np.arange(DR,RMAX+DR,DR)
        self.Atoms = Atoms
        self.legendreOrder = legendreOrder
        self.model     = self.generate_scattering_model()

        # Run initializaiton from parent class
        super(sphericalBesselChiSquare, self).__init__(meanSignal,
                                                 stdSignal, self.model,
                                                 reconType)

    def setBounds(self,boundsFunction):
        self.bounds = boundsFunction(self.Rs)

    def generate_scattering_model(self):
        QQ, RR = np.meshgrid(self.Qs, self.Rs)
        return fQ(QQ,self.Atoms[0])**2*special.spherical_jn(self.legendreOrder, QQ*RR)*RR**2
