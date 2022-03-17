import numpy as np
import astropy.units as u
from astropy.constants import L_sun
from scipy.interpolate import interp1d
from astropy.constants import c

from qlfhosts.util.Lhost_Lagn_max_v2 import Lhost_Lagn_max

"""
This version of the script matches what is done in the Shen20 pubtools, and gives very similar results. However, I think this is actually something I don't follow in their implementation in how Lx is calculated for f(NH; Lx, z).

Additionally, it applies the Mi_lim in L_bol space, which I think it is the more correct thing to do.

"""

def get_Lfrac_lam(Lfrac, Lstar_10, qlf):
    """
    This function returns L_lam(L)/L_lam(Lstar). This function is only valid for UV/optical wavelengths, were we assume the conversion factors are just proportional to the B-band conversion.

    Parameters
    ----------

    Lfrac: numpy array
        Values of L/Lstar for which to calculate Lfrac_lam = L_lam/L_lam(Lstar)

    Lstar_10: float
        Value of Lstar in units of 10^10 Lsun.

    qlf: QLF object
        QLF being used.

    """
    D = np.tile(qlf.c_B*Lstar_10**qlf.k_B, [len(Lfrac),1])
    Lfrac_2D = np.tile(Lfrac, [len(qlf.c_B),1]).T
    return np.sum(D,axis=1)/np.sum(D*Lfrac_2D**(qlf.k_B-1),axis=1)

def jacobian(Lfrac, Lstar_10, qlf):
    """
    This function returns the jacobian dlog L / dlog L_lam.

    Parameters
    ----------

    Lfrac: numpy array
        Values of L/Lstar for which to calculate Lfrac_lam = L_lam/L_lam(Lstar)

    Lstar_10: float
        Value of Lstar in units of 10^10 Lsun.

    qlf: QLF object
        QLF being used.

    """
    D = np.tile(qlf.c_B*Lstar_10**qlf.k_B, [len(Lfrac),1])
    Lfrac_2D = np.tile(Lfrac, [len(qlf.c_B),1]).T
    return np.sum(-D*Lfrac_2D**qlf.k_B,axis=1) / np.sum(D*(qlf.k_B -1)*Lfrac_2D**qlf.k_B,axis=1)
    #return np.sum(D*(1.+qlf.k_B)*Lfrac_2D**qlf.k_B, axis=1)/np.sum(D*Lfrac_2D**qlf.k_B, axis=1)



def get_phi_lam_obs(z, qlf, lLfrac_lam_obs_min, lLfrac_lam_obs_max, lam_eff_filter, agn_sed, hosts_sed, sel_crit, glf, lLfrac_min_lim=None):

    """
    This is the main function of this module. For a given redshift, it returns the observed qlf at a given observed wavelength equal to the effective wavelength of the filter used / (1+z).

    Parameters
    ----------

    z: float
        Redshift.

    qlf: QLF object.
        QLF model being used.

    lLfrac_lam_obs_min: float
        Lower limit of log L_lam_obs / L_lam(Lstar) to produce the observed QLF.

    lLfrac_lam_obs_max: float
        Upper limit of log L_lam_obs / L_lam(Lstar) to produce the observed QLF.

    lam_eff_filter: float
        Effective wavelength of the filter being used.

    """

    #Start by getting the value of Lstar in units of 10^10 Lsun, which will be useful later on.
    Lstar = 10.**(qlf.log_Lstar(z))*qlf.Lstar_units
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
    phi_bol = qlf.phi_bol_Lfrac(Lfrac, z)

    #Apply the limit in requested.
    if lLfrac_min_lim is not None:
        #phi_bol[lLfrac<=lLfrac_min_lim] = 0.
        phi_bol[lLfrac<=lLfrac_min_lim] = 1.e-32*phi_bol.unit

    #Transform the bolometric QLF to the intrinsic luminosity QLF in the band. We assume that the bolometric correction in all bands of interest is proportional to the one in the B-band, as is done in the Hopkins07 provided code.
    phi_lam = phi_bol*jacobian(Lfrac, Lstar_10, qlf)
    Lfrac_lam    = get_Lfrac_lam(Lfrac, Lstar_10, qlf)
    lLfrac_lam   = np.log10(Lfrac_lam)
    #dlLfrac_lam  = dlLfrac/jacobian(Lfrac, Lstar_10, qlf)

    #Since there is a natural dispersion to the bolometric corrections, we convolve phi_lam with the uncertainty function to take it into account.
    phi_lam_2D        = np.tile(phi_lam, (len(phi_lam), 1))
    sigma             = qlf.get_sigma(Lfrac, Lstar_10, lam_eff_filter/(1.+z))
    lLfrac_lam_sig    = lLfrac_lam
    sigma_2D          = np.tile(sigma, (len(sigma), 1))
    lLfrac_lam_2D     = np.tile(lLfrac_lam, (len(lLfrac_lam), 1))
    lLfrac_lam_sig_2D = np.tile(lLfrac_lam_sig, (len(lLfrac_lam), 1)).T

    p = (2.*np.pi)**-0.5 * sigma_2D**-1 * np.exp( -0.5*( (lLfrac_lam_sig_2D - lLfrac_lam_2D)/sigma_2D)**2)

    phi_lam_sig = np.sum(phi_lam_2D*p * dlLfrac, axis=1)

    #The next step is to convolve with the obscuration function. The issue here is that the observed luminosity in the band is a function of the intrinsic luminosity and the obscuration.
    lNH_min = 20.
    lNH_max = 26.
    dlNH    = 0.01
    lNH     = np.arange(lNH_min, lNH_max, dlNH)

    #Following the approach of the Shen20 pubtools, we will now calculate phi_lam_obs for the same luminosity fractions for which we have phi_lam.
    lLfrac_lam_obs_grid = lLfrac_lam_sig

    #Determine the obscuration function in the observed band.
    ltheta_fact = 0.4*qlf.dgr(z).to(u.cm**2).value*1e22 * qlf.xi(lam_eff_filter/(1.+z))
    ltheta = 10.**(lNH-22) * ltheta_fact
    ltheta_2D = np.tile(ltheta, [len(lLfrac_lam_obs_grid), 1])

    #Calculate the maximum host luminosity we can tolerate for each value of the reddening being considered. 
    Lh_La_max = np.zeros((lNH.shape[0],hosts_sed.likelihood.shape[0]))
    print(Lh_La_max.shape)
    input()
    A_lams = 2.5*ltheta
    for k, A_lam in enumerate(A_lams):
        Lh_La_max[k] = Lhost_Lagn_max(agn_sed, hosts_sed, A_lam, lam_eff_filter/(1.+z), sel_crit)
        print(Lh_La_max[k])
        input()
    Lh_La_max_2D = np.tile(Lh_La_max, [len(lLfrac_lam_obs_grid),1])

    nu_rest = (c/(lam_eff_filter/(1.+z))).to(u.Hz)
    L_nu    = qlf.L_at_lam(Lfrac*Lstar, lam_eff_filter/(1.+z))/nu_rest
    L_nu_2D = np.tile(L_nu, [len(lNH), 1]).T
    #print(Lh_La_max_2D.shape, L_nu_2D.shape)
    Lh_nu_max_2D = Lh_La_max_2D * L_nu_2D

    #Lh_nu_max = (Lh_La_max * Lfrac*Lstar / (c/lam_eff_filter)).to(u.erg/u.s/u.Hz).value
    #print(Lh_nu_max_2D)
    #for k in range(len(lNH)):
    #    print("{0:.2f} {1:.5f} {2:.2f}".format(lNH[k], Lh_La_max[k], glf.P(Lh_nu_max[k].to(u.erg/u.s/u.Hz).value)))
    L_bol_AGN = Lfrac*Lstar
    L_bol_AGN_2D = np.tile(L_bol_AGN, [len(lNH), 1]).T
    P_2D = glf.P(Lh_nu_max_2D, sed=agn_sed, L_AGN = L_bol_AGN_2D)
    #print(P_2D)
    # exit()

    #For each NH, we will need to evaluate the unreddened QLF at a luminosity of lLfrac_lam_obs_grid + ltheta. So let's build it as a 2D array in which each row has the same lLfrac_lam_obs_grid value modified by the reddening correction (i.e., unreddened assuming different levels of obscuration).
    lLfrac_lam_sig_eval_2D = np.tile(lLfrac_lam_obs_grid, [len(lNH), 1]).T + ltheta_2D

    #Now, evaluate the f_NH function, following the S20 pubtools. Note: I think this actually wrong. f_NH should be evaluated at the intrinsic luminosity fraction of the reddening corrected luminosity. Here, we just assume that the same intrinsic lLfrac corresponds to the observed lLfrac_lam_obs_grid value for all NHs.
    lLfrac_eval_2D = np.tile(lLfrac, [len(lNH),1]).T
    log_NH_2D = np.tile(lNH, [len(lLfrac_lam_obs_grid), 1])
    f_NH = qlf.fNH(log_NH_2D, lLfrac_eval_2D, Lstar_10, z)

    #Extrapolate phi_lam_sig so that we can evaluate it in the new positions.
    log_phi_lam_sig_interp = interp1d(lLfrac_lam_sig, np.log10(phi_lam_sig.value+1e-32), kind='linear', fill_value = 'extrapolate')

    #Evaluate it and produce phi_lam_obs_grid by integrating over f_NH dlNH.
    phi_lam_sig_eval_2D = 10.**(log_phi_lam_sig_interp(lLfrac_lam_sig_eval_2D))
    phi_lam_obs_grid= np.sum(phi_lam_sig_eval_2D * f_NH * dlNH * P_2D, axis=1)

    #Now, this is the output grid we actually want.
    nlLfrac_lam_obs    = 100
    dlLfrac_lam_obs    = (lLfrac_lam_obs_max-lLfrac_lam_obs_min)/nlLfrac_lam_obs
    if dlLfrac_lam_obs > 0.1:
        dlLfrac_lam_obs    = 0.1
    lLfrac_lam_obs     = np.arange(lLfrac_lam_obs_min, lLfrac_lam_obs_max + 0.1*dlLfrac_lam_obs, dlLfrac_lam_obs)

    #Interpolate/extrapolate phi_lam_obs to put it in the required output grid and return the resulting QLF.
    #print(np.min(lLfrac_lam_obs_grid), np.max(lLfrac_lam_obs_grid))
    lphi_lam_obs_interp = interp1d(lLfrac_lam_obs_grid, np.log10(phi_lam_obs_grid+1e-32), fill_value='extrapolate', kind='cubic')
    phi_lam_obs = 10.**(lphi_lam_obs_interp(lLfrac_lam_obs))*phi_lam_sig.unit
    return phi_lam_obs, dlLfrac_lam_obs*u.dex
