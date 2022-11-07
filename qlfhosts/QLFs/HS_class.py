import numpy as np
from astropy.constants import L_sun, c
import astropy.units as u

class HS_class(object):

    def __init__(self):
        return

    def get_phi_lam_no_red(self, z, lam_rest, Mi_lim=None):

        #Get the luminosity function. 
        phi_lam_sig, lLfrac_lam_sig, lLfrac = self.get_phi_lam_no_red_Lfrac(z, lam_rest)

        #Transform the luminosity fractions into luminosities.
        Lstar = 10.**(self.log_Lstar(z))*self.Lstar_units
        Lstar_lam = self.L_at_lam(Lstar, lam_rest)
        lLlam_sig = lLfrac_lam_sig + np.log10(Lstar_lam.to(self.Lstar_units).value)
        lLbol = lLfrac + self.log_Lstar(z)

        #Place a cut at the given absolute magnitude if requested. 
        if Mi_lim is not None:
            #The log of L_at_iband. 
            Fnu_i_lim = 3631.*u.Jy * 10**(-0.4*Mi_lim)
            L_i_lim = Fnu_i_lim * 4.*np.pi * (10.*u.pc)**2 * (c/(7500.*u.AA))
            lL_i_lim = np.log10(L_i_lim/self.Lstar_units)
            #Use the Lstar_at_lam and Lstar_at_i to convert into L_lam_lim.
            Lstar_i = self.L_at_lam(Lstar, 7500.*u.AA)
            lLlam_lim = lL_i_lim + np.log10(Lstar_lam/Lstar_i)
            #Apply the limit. 
            phi_lam_sig[lLlam_sig<=lLlam_lim] = 0.
            
        #Return the values.
        return phi_lam_sig, lLlam_sig, lLbol

    def get_phi_lam_no_red_Lfrac(self, z, lam_rest, lLfrac_min_lim=None):

        #Start by getting the value of Lstar in units of 10^10 Lsun, which will be useful later on.
        Lstar = 10.**(self.log_Lstar(z))*self.Lstar_units
        Lstar_10 = (Lstar/(1e10*L_sun)).to(1.).value

        #Set the grid in bolometric L/Lstar.
        #lLfrac_min = -3.0
        #lLfrac_max =  3.0 #10.0
        lLfrac_min = -6.0
        lLfrac_max =  6.0
        dlLfrac    =  0.01
        lLfrac     = np.arange(lLfrac_min,lLfrac_max,dlLfrac)
        Lfrac      = 10.**lLfrac

        #Get the bolometric QLF evaluated in the grid of Lfrac.
        phi_bol = self.phi_bol_Lfrac(Lfrac, z)

        #Apply the limit in requested.
        if lLfrac_min_lim is not None:
            #phi_bol[lLfrac<=lLfrac_min_lim] = 0.
            phi_bol[lLfrac<=lLfrac_min_lim] = 1.e-32*phi_bol.unit

        #Transform the bolometric QLF to the intrinsic luminosity QLF in the band. We assume that the bolometric correction in all bands of interest is proportional to the one in the B-band, as is done in the Hopkins07 provided code.
        phi_lam = phi_bol*self.jacobian(Lfrac, Lstar_10)
        Lfrac_lam    = self.get_Lfrac_lam(Lfrac, Lstar_10)
        lLfrac_lam   = np.log10(Lfrac_lam)
        #dlLfrac_lam  = dlLfrac/jacobian(Lfrac, Lstar_10, qlf)

        #Since there is a natural dispersion to the bolometric corrections, we convolve phi_lam with the uncertainty function to take it into account.
        phi_lam_2D        = np.tile(phi_lam, (len(phi_lam), 1))
        sigma             = self.get_sigma(Lfrac, Lstar_10, lam_rest)
        lLfrac_lam_sig    = lLfrac_lam
        sigma_2D          = np.tile(sigma, (len(sigma), 1))
        lLfrac_lam_2D     = np.tile(lLfrac_lam, (len(lLfrac_lam), 1))
        lLfrac_lam_sig_2D = np.tile(lLfrac_lam_sig, (len(lLfrac_lam), 1)).T

        p = (2.*np.pi)**-0.5 * sigma_2D**-1 * np.exp( -0.5*( (lLfrac_lam_sig_2D - lLfrac_lam_2D)/sigma_2D)**2)

        phi_lam_sig = np.sum(phi_lam_2D*p * dlLfrac, axis=1)

        return phi_lam_sig, lLfrac_lam_sig, lLfrac


    def get_Lfrac_lam(self, Lfrac, Lstar_10):
        """
        This function returns L_lam(L)/L_lam(Lstar). This function is only valid for UV/optical wavelengths, were we assume the conversion factors are just proportional to the B-band conversion.

        Parameters
        ----------

        Lfrac: numpy array
            Values of L/Lstar for which to calculate Lfrac_lam = L_lam/L_lam(Lstar)

        Lstar_10: float
            Value of Lstar in units of 10^10 Lsun.

        """
        D = np.tile(self.c_B*Lstar_10**self.k_B, [len(Lfrac),1])
        Lfrac_2D = np.tile(Lfrac, [len(self.c_B),1]).T
        return np.sum(D,axis=1)/np.sum(D*Lfrac_2D**(self.k_B-1),axis=1)

    def jacobian(self, Lfrac, Lstar_10):
        """
        This function returns the jacobian dlog L / dlog L_lam.

        Parameters
        ----------

        Lfrac: numpy array
            Values of L/Lstar for which to calculate Lfrac_lam = L_lam/L_lam(Lstar)

        Lstar_10: float
            Value of Lstar in units of 10^10 Lsun.

        """
        D = np.tile(self.c_B*Lstar_10**self.k_B, [len(Lfrac),1])
        Lfrac_2D = np.tile(Lfrac, [len(self.c_B),1]).T
        return np.sum(-D*Lfrac_2D**self.k_B,axis=1) / np.sum(D*(self.k_B -1)*Lfrac_2D**self.k_B,axis=1)
        #return np.sum(D*(1.+qlf.k_B)*Lfrac_2D**qlf.k_B, axis=1)/np.sum(D*Lfrac_2D**qlf.k_B, axis=1)
