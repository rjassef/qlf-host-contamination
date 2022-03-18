from matplotlib.pyplot import hot
import numpy as np
import astropy.units as u
from astropy.constants import L_sun, c
from scipy.special import erf

class Kollmeier06(object):

    def __init__(self):

        self.mu = np.log10(0.25)
        self.sigma = 0.3

        return

    def P(self, Lh_nu_max, **kargs):

        #Chek that the SED model object was passed along.
        for key in ('L_AGN', 'hosts_sed', 'k_hosts_sed', 'lam_rest'):
            if key not in kargs:
                print('To calculate P in the Kollmeier06 module you need to include the parameter {}.'.format(key))
                return [None]*len(Lh_nu_max)

        #Get the corresponding V-band bulge luminosity density for the host luminosity density at the rest-frame wavelength of the observations.
        hosts_sed = kargs['hosts_sed']
        k_hosts_sed = kargs['k_hosts_sed']
        lam_rest = kargs['lam_rest']
        Lh_nu_V_max = Lh_nu_max * hosts_sed.Lnu(0.545*u.micron)[k_hosts_sed]/hosts_sed.Lnu(lam_rest)[k_hosts_sed] * (lam_rest/(0.545*u.micron)).to(1).value
        Lh_nu_V_bulge_max = hosts_sed.bt_ratio[k_hosts_sed] * Lh_nu_V_max

        #Now, get the SMBH mass using the relation in eqn. (5) of Gultekin+09. We assume the solar Johnson-V luminosity density of Willmer (2018). Note that we get M_BH in solar masses.
        M_V_sun_AB = 4.80
        Lnu_V_sun = 3631.*u.Jy * 10**(-0.4*M_V_sun_AB) * 4*np.pi*(10*u.pc)**2
        lM_BH_max = 8.95 + 1.11*np.log10(Lh_nu_V_bulge_max/(1e11*Lnu_V_sun) + 1e-32)

        #Determine lambda_min, the minimum Eddington ratio possible. 
        l_Ledd_max = np.log10(3.2)+4 + lM_BH_max #In solar luminosities.
        L_AGN = kargs['L_AGN']
        l_lambda_min = np.log10(L_AGN/L_sun) - l_Ledd_max

        #This is a convenience variable for the integration.
        xmin = (l_lambda_min - self.mu)/(self.sigma * 2**0.5)
        Prob = np.where(Lh_nu_max > 0., 
            0.5*(1-xmin/np.abs(xmin) * erf(np.abs(xmin))), 
            0.)

        #Return the probability.
        return Prob.value
