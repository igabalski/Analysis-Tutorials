import os

import sys
sys.path.insert(0, os.environ['dropboxPath']+'/Code/Lab/TRXS/Libraries/trxsToolBox/formFactors')
from formFactors import *

from scipy import special

def generateFourierGrid( R ):
    DR   = (R[1]-R[0])
    QMAX = np.pi / DR
    DQ   = 2*np.pi/(R.max()-R.min())
    Q    = np.arange(0,QMAX+DQ,DQ)
    return Q

def fromRho( rho, R, legendreOrder = 0 , atom = 'I', withFormFactor=True ):
    DR   = (R[1]-R[0])
    Q    = generateFourierGrid( R )


    if withFormFactor:
        Qmin,Qmax = fQ( Q, atom, return_bounds=True )
        Q  = Q[ ( Q>Qmin )&( Q<Qmax )]

    [ RR , QQ ] = np.meshgrid( R , Q )
    # [ QQ , RR ] = np.meshgrid( Q , R )
    if withFormFactor:
        ff = fQ( Q, atom )

    integrand = special.spherical_jn(legendreOrder, QQ*RR)*RR**2*DR
    rhoN = rho / np.dot( R**2*DR , rho )
    S = 1 + np.dot( integrand , rhoN )

    if withFormFactor:
        return {'Q':Q,'S': ff**2*S }
    else:
        return {'Q':Q,'S': S }

def fromPerfectlyAlignedDiatomic( rho, y, atom='Iodine', withFormFactor=True ):
    """
        perfectlyAlignedDiatomic returns the scattering along y.
        The scattering signal will vary along x due to the form factor.

        Input:
            rho: Diatomic probability density
            y:   The grid along which rho is defined
            atom: The atoms to use to model scattering
            withFormFactor: toggles the form factor on/off
        Output:
            Dictionary {'Qy':Qy,'S':S}
    """

    DY   = (y[1]-y[0])
    QY   = generateFourierGrid( y )


    if withFormFactor:
        Qmin,Qmax = fQ( QY, atom, return_bounds=True )
        QY = QY[ ( QY>Qmin )&( QY<Qmax )]

    [ YY , QQ ] = np.meshgrid( y , QY )
    # [ QQ , RR ] = np.meshgrid( Q , R )
    if withFormFactor:
        ff = fQ( QY, atom )

    integrand = np.cos(QQ*YY)*DY
    rhoN = rho / np.sum( DY * rho )
    S = 1 + np.dot( integrand , rhoN )

    if withFormFactor:
        return {'Qy':QY,'S': ff**2*S }
    else:
        return {'Qy':QY,'S': S }

################################################################################
#~~~~~~~~~Generate scattering angles from x,y,L detector positions
################################################################################
def generateThetaPhi( x, y, L ):
    theta = np.arctan2(x, y)
    phi   = np.arctan2(L, np.sqrt(x**2+y**2)  )
    return ( theta, phi )


################################################################################
#~~~~~~~~~Generate Q from theta, phi, wavelength
################################################################################
def generateQ( theta, phi, wavelength ):
    k0 = 2*np.pi/wavelength
    Qx = k0*( np.cos(phi)*np.sin(theta) )
    Qy = k0*( np.sin(phi)*np.sin(theta) )
    Qz = k0*( np.cos(theta) - 1 )
    Q  = np.sqrt(Qx**2 + Qy**2 + Qz**2)

    return {'Q':Q,'Qvec':(Qx,Qy,Qz)}


################################################################################
#~~~~~~~~~Thomson scattering
################################################################################
def thomsonScatteringCS( theta, phi, polarization_angle=0 ):
    print "thomsonScatteringCS needs to be updated with the correct term"
    return np.cos(theta+polarization_angle)**2 + np.sin(theta+polarization_angle)**2 * np.cos(phi)**2


################################################################################
#~~~~~~~~~2D scattering
################################################################################
def fromPerfectlyAlignedDiatomic2D( rho, y, theta, phi, wavelength,
                                    atom='Iodine', withFormFactor=True,
                                    withThomsonCS=True, polarization_angle=0 ):

    DY   = (y[1]-y[0])
    qDict = generateQ( theta,phi,wavelength )
    QY   = qDict['Qvec'][1].flatten()

    [ YY , QQ ] = np.meshgrid( y , QY )

    ff = np.ones_like( theta )
    thCS = np.ones_like( theta )
    if withFormFactor:
        ff = fQ( qDict['Q'], atom )
    if withThomsonCS:
        thCS = thomsonScatteringCS( theta, phi, polarization_angle=polarization_angle )


    integrand = np.cos(QQ*YY)*DY
    rhoN = rho / np.sum( DY * rho )
    S = np.reshape(1 + np.dot( integrand , rhoN ), theta.shape, order='C')

    return thCS*ff**2*S
