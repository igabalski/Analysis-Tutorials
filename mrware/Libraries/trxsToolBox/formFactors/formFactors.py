import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
# Reference
# http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# or http://it.iucr.org/Cb/ch6o1v0001/.
def fAtom( Qs, a, b, c ):
    f1 = a[0]*np.exp(-b[0]*(Qs/(4*np.pi))**2)
    f2 = a[1]*np.exp(-b[1]*(Qs/(4*np.pi))**2)
    f3 = a[2]*np.exp(-b[2]*(Qs/(4*np.pi))**2)
    f4 = a[3]*np.exp(-b[3]*(Qs/(4*np.pi))**2)
    return f1+f2+f3+f4+c

def fQ(Qs,Atom):
    """
    Returns the scattering form factor at the momentum transfer coordinates
    given by Qs (inverse Angstroms) for the specified atom.*

    Args:
        Qs: The grid of momentum transfer coordinates in inverse Angstroms
        Atom: String specifying the atom of interest

    *You'll need to add the definition of f(Q) for the atoms in your molecule

    Returns:
        The scattering form factor
    """
    if Atom == 'Iodine':
        qPaper = np.concatenate([np.arange(0,3.1e-2,5e-3), [4e-2], np.arange(5,11,2)*1e-2, np.arange(1,2.25,.25)*1e-1, [2.5e-1],
                  np.arange(3,11,1)*1e-1, [1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10]],axis=0);

        fPaper = np.array([53, 5.2986e1, 5.2952e1, 5.2895e1, 5.2814e1, 5.2711e1, 5.2586e1, 5.2273e1, 5.1884e1, 5.09e1, \
                           4.9707e1, 4.9057e1, 4.7335e1, 4.5568e1, 4.3842e1, \
                      4.2195e1, 3.9199e1, 3.6555e1, 3.1950e1, 2.7973e1, 2.4636e1, 2.199e1, 1.9956e1, 1.8374e1, 1.7076e1, \
                      1.4354e1, 1.1867e1, 7.9779, 5.9245, 5.0953, 4.2662, 3.7244, 2.6408, 1.8841, 1.471, 1.3138, 9.973e-1]);

        f = interp1d(qPaper,fPaper, kind='cubic')
        fQ = f(Qs)
    elif Atom == 'Carbon':
        a = [2.31, 1.02, 1.5886, 0.865]
        b = [20.8439, 10.2075, .5687, 51.6512]
        c = .2156

        fQ = fAtom( Qs, a, b, c )
    elif Atom == 'Sulfur':
        a = [6.9053,5.2034,1.4379,1.5863]
        b = [1.4679,22.2151,.2536,56.172]
        c = .8669

        fQ = fAtom( Qs, a, b, c )
    elif Atom == 'Nitrogen':
        a = [12.2126,3.1322,2.0125,1.1663]
        b = [0.0057,9.8933, 28.9975, 0.5826]
        c = -11.529
        fQ = fAtom( Qs, a, b, c )
    elif Atom == 'Oxygen':
        a = [3.0485,2.2868,1.5463,0.867]
        b = [13.2771,5.7011,0.3239,32.9089]
        c = 0.2508
        fQ = fAtom( Qs, a, b, c )
    else:
        raise NameError('Your atom is not defined in function fQ of this library.')

    return fQ
