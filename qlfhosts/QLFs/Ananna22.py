import numpy as np
from astropy.constants import c, L_sun
import astropy.units as u
from scipy.interpolate import interp1d
import warnings
from astropy.table import Table
import os

from .Pei92 import P92_Extinction

class QLF(object):

    '''
    Do not use this function. It has the QLF for z=0, but does not evolve with redshift.
    
    '''

    def __init__(self):

        self.kappa = 7.4

        self.log_Lx_star = 44.17
        self.log_Lbol_star = self.log_Lx_star + np.log10(self.kappa)
        self.Lstar_units = u.erg/u.s
        self.Lbol_star = 10**self.log_Lbol_star * self.Lstar_units

        self.log_phi_star = -4.71
        self.h = 0.7
        self.phi_star_units = u.dex**-1 * (self.h)**3 * u.Mpc**-3
        self.phi_star = 10**self.log_phi_star * self.phi_star_units

        self.gamma1 = 0.75
        self.eps_gamma = 1.64

        #FROM HERE BELOW, COPY FROM SHEN20

        #Coefficients to calculate the bolometric correction for B-band. Using the model implemented in Shen et al. (2020)
        self.c_B = np.array([3.759, 9.830])
        self.k_B = np.array([-0.361, -0.0063])

        #Set the reddening model.
        self.red_model = P92_Extinction("MW")

        #Read the Richards et al. (2006) mean quasar SED and generate an interpolation function. 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',u.UnitsWarning) 
            R06_SED_tab = Table.read(os.path.dirname(__file__)+"/Richards_06.dat", format='ascii.cds')
        lam_R06_SED = (c/(10**(R06_SED_tab['LogF'].data)*u.Hz)).to(u.micron).value
        self.lL_R06_SED  = interp1d(lam_R06_SED, R06_SED_tab['All'].data)

        #Save the value at B-band (4400A)
        self.lL_R06_at_B = self.lL_R06_SED(0.44)

        #Dust to gas ratio assumed: A_B/NH
        self.dgr_local = 8.47e-22 * u.cm**2


        return

    def get_phi_lam_no_red(self, z, lam_rest, Mi_lim=None):

        lLfrac_min = -6.0
        lLfrac_max =  6.0
        dlLfrac    =  0.01
        lLfrac     = np.arange(lLfrac_min,lLfrac_max,dlLfrac)
        Lfrac      = 10.**lLfrac

        phi =  self.phi_star * Lfrac**(-self.gamma1) / (1 + Lfrac**self.eps_gamma)

        lLbol  = lLfrac + self.log_Lbol_star
        Llam  = self.L_at_lam(10**(lLbol) * self.Lstar_units, lam_rest)
        lLlam = np.log10(Llam/self.Lstar_units)

        return phi, lLlam, lLbol

    def log_Lstar(self, z):
        return self.log_Lbol_star


    def L_at_lam(self, Lbol, lam):
        """
        This method returns the luminosity of a quasar of bolometric luminosity Lbol by scaling the B-band luminosity using the Richard et al. (2006) mean quasar SED. 
        """

        #Get the B-band luminosity. 
        L_at_B = self.L_B(Lbol)

        #Get the ratio in mean quasar SED between L_B and L at lambda. 
        lL_ratio = self.lL_R06_SED(lam.to(u.micron).value) - self.lL_R06_at_B

        #Return the luminosity at wavelength lambda. 
        return L_at_B * 10.**(lL_ratio)

    def L_B(self, Lbol):
        """
        This method returns the B-band luminosity of a quasar of bolometric luminosity Lbol using equation (5) of S20.

        While the typical use of equation (5) is to determine Lbol given an observable monochromatic luminosity, here we use the conversion to go from Lbol to LB. A direct application of this function is used in the accompanying script mstar.vandenberk.py, where we want to estimate the observed fluxes of a type 1 quasar with bolometric luminosity equal to L*.

        """
        #Implementation of equation (5).
        x = Lbol/(1e10*L_sun)
        bc = self.c_B[0]*x**self.k_B[0] + self.c_B[1]*x**self.k_B[1]
        return Lbol/bc

    def L_x_Lfrac(self, lLfrac):

        lLx_44 = lLfrac + self.log_Lbol_star - 43.75 - np.log10((self.Lstar_units/(u.erg/u.s)).to(u.dimensionless_unscaled))

        return lLx_44

    def fNH(self, log_NH_2D, lLfrac, Lstar_10=None, z=None):

        #Get the hard x-ray luminosity for each Lfrac in units of 10^44 erg/s. This will be useful later.
        lLfrac_use = np.where(lLfrac>10.0, 10., lLfrac)
        lLfrac_use = np.where(lLfrac_use<-10.0, -10., lLfrac_use)
        lLx_44 = self.L_x_Lfrac(lLfrac_use)
        lLx_44 = np.where(lLfrac >  10,  np.inf, lLx_44)
        lLx_44 = np.where(lLfrac < -10, -np.inf, lLx_44)

        f_CTK = 1.0
        eps = 1.7
        psi_max = 0.84
        psi_min = 0.2
        if z<2.0:
            psi44 = 0.43*(1+z)**0.48
        else:
            psi44 = 0.43*(1+2)**0.48
        psi = psi44 - 0.24*lLx_44
        psi = np.where(psi<psi_min, psi_min, psi)
        psi = np.where(psi>psi_max, psi_max, psi)

        f_20_21_1 = 1.0 - (2.0+eps)/(1.0+eps) * psi
        f_21_22_1 = 1.0/(1.0+eps) * psi
        f_22_23_1 = 1.0/(1.0+eps) * psi
        f_23_24_1 = eps/(1.0+eps) * psi
        f_24_26_1 = f_CTK/2.0 * psi

        f_20_21_2 = 2.0/3.0 - (3.0+2.0*eps)/(3.0+3.0*eps) * psi
        f_21_22_2 = 1.0/3.0 - eps/(3.0+3.0*eps) * psi
        f_22_23_2 = 1.0/(1.0+eps) * psi
        f_23_24_2 = eps/(1.0+eps) * psi
        f_24_26_2 = f_CTK/2.0 * psi

        psi_lim = (1.0+eps)/(3.0+eps)
        f_20_21 = np.where(psi<psi_lim, f_20_21_1, f_20_21_2)
        f_21_22 = np.where(psi<psi_lim, f_21_22_1, f_21_22_2)
        f_22_23 = np.where(psi<psi_lim, f_22_23_1, f_22_23_2)
        f_23_24 = np.where(psi<psi_lim, f_23_24_1, f_23_24_2)
        f_24_26 = np.where(psi<psi_lim, f_24_26_1, f_24_26_2)

        f_20_21 /= (1.0+f_CTK*psi)
        f_21_22 /= (1.0+f_CTK*psi)
        f_22_23 /= (1.0+f_CTK*psi)
        f_23_24 /= (1.0+f_CTK*psi)
        f_24_26 /= (1.0+f_CTK*psi)

        # log_NH_2D = np.tile(log_NH, [len(lLx_44),1]).T
        f_NH = np.zeros(log_NH_2D.shape)
        f_NH = np.where((log_NH_2D>=20.0) & (log_NH_2D<21.0) , f_20_21, f_NH)
        f_NH = np.where((log_NH_2D>=21.0) & (log_NH_2D<22.0) , f_21_22, f_NH)
        f_NH = np.where((log_NH_2D>=22.0) & (log_NH_2D<23.0) , f_22_23, f_NH)
        f_NH = np.where((log_NH_2D>=23.0) & (log_NH_2D<24.0) , f_23_24, f_NH)
        f_NH = np.where((log_NH_2D>=24.0) & (log_NH_2D<=26.0), f_24_26, f_NH)

        return f_NH

    def dgr(self, z):
        return  self.dgr_local * 10.**(0.35 + 0.93*np.exp(-0.43*z)-1.05)/10.**(0.35+0.93-1.05)

    def xi(self, lam):
        return self.red_model.xi_fit(lam)

