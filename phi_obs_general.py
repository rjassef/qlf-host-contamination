import numpy as np
import astropy.units as u
from astropy.constants import L_sun
from scipy.interpolate import interp1d
from astropy.constants import c

from qlfhosts.util.Lhost_Lagn_max import Lhost_Lagn_max

def get_phi_lam_obs(z, qlf, lLlam_obs_min, lLlam_obs_max, lam_eff_filter, agn_sed, hosts_sed, sel_crit, glf, lLfrac_min_lim=None):

    #Effective rest-frame wavelength of the filter.
    lam_eff_rest = lam_eff_filter/(1+z)

    #Get the reddening corrected luminosity function at the rest-frame wavelength mapped by the filter. 
    phi_lam_sig, lLlam_sig, lLbol = qlf.get_phi_lam_no_red(z, lam_eff_rest)

    #The next step is to convolve with the obscuration function. The issue here is that the observed luminosity in the band is a function of the intrinsic luminosity and the obscuration.
    lNH_min = 20.
    lNH_max = 26.
    dlNH    = 0.01
    lNH     = np.arange(lNH_min, lNH_max, dlNH)

    #Following the approach of the Shen20 pubtools, we will now calculate phi_lam_obs for the same luminosity fractions for which we have phi_lam.
    lLlam_obs_grid = lLlam_sig

    #Determine the obscuration function in the observed band.
    ltheta_fact = 0.4*qlf.dgr(z).to(u.cm**2).value*1e22 * qlf.xi(lam_eff_filter/(1.+z))
    ltheta = 10.**(lNH-22) * ltheta_fact
    ltheta_2D = np.tile(ltheta, [len(lLlam_obs_grid), 1])

    #Before calculating the maximum host luminosity allowed, let's pre-compute a number of useful quantities. 
    # - Find the unscaled luminosity ratio. 
    Lnu_agn_unscaled = agn_sed.Lnu(lam_eff_filter/(1.+z))
    Lnu_hosts_unscaled = hosts_sed.Lnu(lam_eff_filter/(1.+z))
    x_unscaled_all = Lnu_hosts_unscaled/Lnu_agn_unscaled[0]
    # - Figure out the reddening factors for the AGN. 
    Aband_to_Alam = dict()
    klam_norm = agn_sed.klam(lam_eff_filter/(1.+z))
    for j, bp in enumerate(agn_sed.bps):
        lam_eff_band = bp.avgwave()
        bp_name = agn_sed.bp_names[j]
        Aband_to_Alam[bp_name] = (agn_sed.klam(lam_eff_band/(1+agn_sed.z)) / klam_norm)

    #Calculate the maximum host luminosity we can tolerate for each value of the reddening being considered for each host template. 
    Lh_La_max = np.zeros((lNH.shape[0],hosts_sed.likelihood.shape[0]))
    A_lams = 2.5*ltheta
    for k, A_lam in enumerate(A_lams):
        Lh_La_max[k] = Lhost_Lagn_max(agn_sed, hosts_sed, A_lam, Aband_to_Alam, sel_crit, x_unscaled_all)

    #Estimate some useful quantities.
    #Lfrac = 10**lLfrac
    nu_rest = (c/(lam_eff_filter/(1.+z))).to(u.Hz)
    L_nu    = (10**lLlam_obs_grid)/nu_rest * qlf.Lstar_units
    L_nu_2D = np.tile(L_nu, [len(lNH), 1]).T
    #L_bol_AGN = Lfrac*Lstar
    L_bol_AGN = 10**(lLbol) * qlf.Lstar_units
    L_bol_AGN_2D = np.tile(L_bol_AGN, [len(lNH), 1]).T

    #This is the array that will hold the combined probabilities. 
    P_2D = np.zeros(L_bol_AGN_2D.shape)

    #Iterate by template to fill out the P_2D array.
    for k_sed in range(len(hosts_sed.sps)):
        if hosts_sed.likelihood[k_sed] == 0.:
            continue
        Lh_La_max_2D = np.tile(Lh_La_max[:,k_sed], [len(lLlam_obs_grid),1])
        Lh_nu_max_2D = Lh_La_max_2D * L_nu_2D

        P_host = glf.P(Lh_nu_max_2D, hosts_sed=hosts_sed, k_hosts_sed=k_sed, L_AGN=L_bol_AGN_2D, lam_rest=lam_eff_filter/(1.+z))
        P_2D += hosts_sed.likelihood[k_sed] * P_host

    #For each NH, we will need to evaluate the unreddened QLF at a luminosity of lLfrac_lam_obs_grid + ltheta. So let's build it as a 2D array in which each row has the same lLfrac_lam_obs_grid value modified by the reddening correction (i.e., unreddened assuming different levels of obscuration).
    lLfrac_lam_sig_eval_2D = np.tile(lLlam_obs_grid, [len(lNH), 1]).T + ltheta_2D

    #Now, evaluate the f_NH function, following the S20 pubtools. Note: I think this actually wrong. f_NH should be evaluated at the intrinsic luminosity fraction of the reddening corrected luminosity. Here, we just assume that the same intrinsic lLfrac corresponds to the observed lLfrac_lam_obs_grid value for all NHs.
    lLfrac = lLbol - qlf.log_Lstar(z)
    lLfrac_eval_2D = np.tile(lLfrac, [len(lNH),1]).T
    log_NH_2D = np.tile(lNH, [len(lLlam_obs_grid), 1])
    Lstar = 10.**(qlf.log_Lstar(z))*qlf.Lstar_units
    Lstar_10 = (Lstar/(1e10*L_sun)).to(1.).value
    f_NH = qlf.fNH(log_NH_2D, lLfrac_eval_2D, Lstar_10, z)

    #Extrapolate phi_lam_sig so that we can evaluate it in the new positions.
    log_phi_lam_sig_interp = interp1d(lLlam_sig, np.log10(phi_lam_sig.value+1e-32), kind='linear', fill_value = 'extrapolate')

    #Evaluate it and produce phi_lam_obs_grid by integrating over f_NH dlNH.
    phi_lam_sig_eval_2D = 10.**(log_phi_lam_sig_interp(lLfrac_lam_sig_eval_2D))
    phi_lam_obs_grid= np.sum(phi_lam_sig_eval_2D * f_NH * dlNH * P_2D, axis=1)

    #Now, this is the output grid we actually want.
    nlLlam_obs    = 100
    dlLlam_obs    = (lLlam_obs_max-lLlam_obs_min)/nlLlam_obs
    if dlLlam_obs > 0.1:
        dlLlam_obs    = 0.1
    lLfrac_lam_obs     = np.arange(lLlam_obs_min, lLlam_obs_max + 0.1*dlLlam_obs, dlLlam_obs)

    #Interpolate/extrapolate phi_lam_obs to put it in the required output grid and return the resulting QLF.
    #print(np.min(lLfrac_lam_obs_grid), np.max(lLfrac_lam_obs_grid))
    lphi_lam_obs_interp = interp1d(lLlam_obs_grid, np.log10(phi_lam_obs_grid+1e-32), fill_value='extrapolate', kind='cubic')
    phi_lam_obs = 10.**(lphi_lam_obs_interp(lLfrac_lam_obs))*phi_lam_sig.unit
    return phi_lam_obs, dlLlam_obs*u.dex
