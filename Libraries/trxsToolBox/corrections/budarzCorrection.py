import pandas as pd
import numpy as np

import sys
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def findPath(name):
    for path in sys.path:
        thePath = find(name,path)
        if thePath is not None:
            return thePath

def budarzCorrection(S,Q,wavelength):
    """
    Correction for the XPP gas-phase scattering cell from the Budarz manuscript.

    Args:
        S: Measured scattered intensity at the points Q
        Q: Momentum transfer in inverse Angstroms
        wavelength: wavelength at which S was measured

    Returns:
        The corrected data
    """
    # Apply Budarz  correction
    df=pd.read_csv(findPath('attenuation_data_8p3keV.csv'))

    angles =  df['Momentum transfer']*1.4938
    qs     =  angles / wavelength

    return S / np.interp(Q, qs, df['Attenuated intensity'])
