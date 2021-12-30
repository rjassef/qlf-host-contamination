import numpy as np
from .schechterFunction import SchechterFunction
import astropy.units as u
from scipy.interpolate import interp1d

#DEEP2 B-band galaxy luminosity function from Willmer et al. (2006)
class Willmer06(SchechterFunction):

    def __init__(self, z):

        #Save the input redshift. 
        self.z = z

        #Same alpha at all redshifts.
        self.alpha  = -1.3

        #For Mstar we will interpolate. 
        z_table = np.array([0.30, 0.50, 0.70, 0.90, 1.10])
        MstarB_table = np.array([-21.07, -21.15, -21.51, -21.36, -21.54])
        if z<=np.min(z_table):
            self.MstarB = MstarB_table[0]
        elif z>=np.max(z_table):
            self.MstarB = MstarB_table[-1]
        else:
            MstarB_func = interp1d(z_table, MstarB_table, kind='linear')
            self.MstarB = MstarB_func(self.z)
        
        #Convert into AB using the conversion in Table 1 of Willmer et al.
        self.MstarB -= 0.10

        #Finally, transform MstarB into Lstar.
        #self.Lstar = 4.*np.pi*(10*u.pc)**2 * 3631*u.Jy * 10**(-0.4*self.MstarB)

        #In erg/s/Hz.
        self.Lstar = 4.34447401e+20 * 10**(-0.4*self.MstarB)

        #Initiate the parent class. 
        super(Willmer06, self).__init__(self.alpha, self.Lstar)

        return
