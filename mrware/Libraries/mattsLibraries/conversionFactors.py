import numpy as np

class conversionFactors():
    def __init__(self):
        # %% Metric constants
        self.c_M=3.e8;
        self.h_M=6.626e-34;
        self.kB_M=1.3806488e-23;
        self.e0_M=8.854e-12;
        self.me_M=9.109e-31;
        self.mH_M=1.672621e-27;

        # %% Hartree atomic units
        self.hbar_AU=1.;
        self.h_AU=2.*np.pi;
        self.me_AU=1.;
        self.kC_AU=1.; # % 1/(4*np.pi e0)
        self.e0_AU=1./4./np.pi;
        self.e_AU=1.;
        self.c_AU=137.;

        # %% Conversions from metric to au (and some other useful conversions)
        self.M_TO_AU=1/0.529e-10;
        self.ANGSTROMS_TO_AU=1/0.529;
        self.EV_TO_AU=1/27.211;
        self.J_TO_AU=1/4.35974417e-18;
        self.D_TO_AU=1/2.541746;
        self.INVCM_TO_AU=100/self.M_TO_AU*self.c_AU*self.h_AU;
        self.KG_TO_AU=1/self.me_M;
        self.AMU_TO_AU=self.mH_M*self.KG_TO_AU;
        self.S_TO_AU=1/2.418e-17;
        self.EF_TO_AU=1/5.14220652e11; # % V/m->au
        self.I_TO_AU=self.J_TO_AU/self.S_TO_AU/(self.M_TO_AU)**2*1e4;# %W/cm**2->au

        self.INTENSITY_TO_AMPLITUDE_M= lambda I: 1.e2*np.sqrt((2.0/self.c_M/self.e0_M)*I);# % W/cm**2->V/m
        self.INTENSITY_TO_AMPLITUDE_AU= lambda I: np.sqrt((2.0/self.c_AU/self.e0_AU)*I);# % I_au->EF_au

    def lengthToMetric(self, atype):
        if atype=='Metric':
            return 1
        elif atype=='Angstroms':
            return 1e-10
        elif atype=='Atomic':
            return 1.0/self.M_TO_AU
        else:
            raise TypeError(atype+' is not a supported unit of length.')

    def massToMetric(self, atype):
        if atype=='Metric':
            return 1
        elif atype=='Atomic':
            return 1/self.KG_TO_AU
        elif atype=='AMU':
            return self.mH_M
        else:
            raise TypeError(atype+' is not a supported unit of mass.')

    def timeToMetric(self, atype):
        if atype=='Metric':
            return 1
        elif atype=='Atomic':
            return 1/self.S_TO_AU
        else:
            raise TypeError(atype+' is not a supported unit of time.')

    def energyToMetric(self, atype):
        if atype=='Metric':
            return 1 # J
        elif atype=='Atomic':
            return 1/self.J_TO_AU
        elif atype=='eV':
            return self.EV_TO_AU/self.J_TO_AU
        elif atype=='iCM':
            return self.INVCM_TO_AU/self.J_TO_AU
        else:
            raise TypeError(atype+' is not a supported unit of energy.')

    def dipoleToMetric(self, atype):
        if atype=='Metric':
            return 1 # C m
        elif atype=='CGS':
            return 3.33564e-30
        elif atype=='Atomic':
            return 3.33564e-30/0.393430307
        else:
            raise TypeError(atype+' is not a supported unit of the electric dipole.')

    def efieldToMetric(self, atype):
        if atype=='Metric':
            return 1 # V/m
        elif atype=='Atomic':
            return 1/self.EF_TO_AU
        else:
            raise TypeError(atype+' is not a supported unit of the electric field.')

    def intensityToMetric(self, atype):
        if atype=='Metric':
            return 1 # W/cm**2
        elif atype=='Atomic':
            return 1/self.I_TO_AU
        else:
            raise TypeError(atype+' is not a supported unit of the field intensity.')


    def measure(self, measure, value, typeI, typeF):
        if (typeI == 'Metric'):
            return value/self.measureToMetric(measure,typeF)
        elif (typeF == 'Metric'):
            return value*self.measureToMetric(measure,typeI)
        else:
            return value*self.measureToMetric(measure,typeI)/self.measureToMetric(measure,typeF)

    def measureToMetric(self, measure, atype):
        if measure == 'Length':
            return self.lengthToMetric(atype)
        elif measure == 'Mass':
            return self.massToMetric(atype)
        elif measure == 'Time':
            return self.timeToMetric(atype)
        elif measure == 'Energy':
            return self.energyToMetric(atype)
        elif measure == 'Dipole':
            return self.dipoleToMetric(atype)
        elif measure == 'eField':
            return self.efieldToMetric(atype)
        elif measure == 'Intensity':
            return self.intensityToMetric(atype)
        else:
            raise ValueError(measure+' is not a supported measure.')
