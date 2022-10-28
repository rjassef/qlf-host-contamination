import numpy as np
import astropy.units as u
from astropy.constants import L_sun
from scipy.interpolate import interp1d
from astropy.constants import c

from .Lhost_Lagn_max import Lhost_Lagn_max

class PhiObs(object):

    def __init__(self, z, QLF=None, Selection=None, AGN_SED=None, Host_SEDs=None, Host_SED_likelihood=None, Galaxy_Luminosity_Distribution=None, cosmo=None):

        #Save all the function call arguments.
        save_args = locals()
        del save_args['self']

        #Save the redshift.
        self.z = z

        #If the cosmology is not provided, set it up here. 
        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        self.cosmo = cosmo

        #Initialize the qlf to use.
        if QLF is None:
            from ..QLFs.Shen20 import QLF
        self.qlf = QLF()

        #Initialize the selection criterion.
        if Selection is None:
            from ..AGN_Selection.R02 import R02 as Selection
        self.sel_crit = Selection()

        #Now, initialize the AGN SED model. The model has to be initialized with the selection function as we need to precompute the colors in the relevant filters for the selection function. 
        if AGN_SED is None:
            from ..SEDs.R06_AGN import R06_AGN as AGN_SED
        self.agn_sed = AGN_SED(self.z, bp_names=self.sel_crit.bp_names, cosmo=cosmo)

        #Initialize the Host SEDs. Again, here we need the selection function to load the proper filters and compute the relevant colors.
        if Host_SEDs is None:
            from ..SEDs.A10_hosts import A10_hosts as Host_SEDs
        self.host_seds = Host_SEDs(self.z, bp_names=self.sel_crit.bp_names, cosmo=cosmo)
        if Host_SED_likelihood is None:
            self.host_seds.likelihood = np.array([0. ,1. ,0.])
        else:
            self.host_seds.likelihood = Host_SED_likelihood

        #If no Galaxy Luminosity Distribution Model is initialized, simply use the Uniform model so that there are no host galaxy effects.
        if Galaxy_Luminosity_Distribution is None:
            from ..GLFs.Uniform import Uniform as Galaxy_Luminosity_Distribution
        self.glf = Galaxy_Luminosity_Distribution(**save_args)

        return

    def get_phi_lam_obs(self, lLlam_obs_min, lLlam_obs_max, lam_eff_filter, lLfrac_min_lim=None):

        #Effective rest-frame wavelength of the filter.
        lam_eff_rest = lam_eff_filter/(1+self.z)

        #Get the reddening corrected luminosity function at the rest-frame wavelength mapped by the filter. 
        phi_lam_sig, lLlam_sig, lLbol = self.qlf.get_phi_lam_no_red(self.z, lam_eff_rest)

        #Get the obscuration. 

        #The next step is to convolve with the obscuration function. The issue here is that the observed luminosity in the band is a function of the intrinsic luminosity and the obscuration.
        lNH_min = 20.
        lNH_max = 26.
        dlNH    = 0.01
        lNH     = np.arange(lNH_min, lNH_max, dlNH)

        #Following the approach of the Shen20 pubtools, we will now calculate phi_lam_obs for the same luminosity fractions for which we have phi_lam.
        lLlam_obs_grid = lLlam_sig

        #Determine the obscuration function in the observed band.
        ltheta_fact = 0.4*self.qlf.dgr(self.z).to(u.cm**2).value*1e22 * self.qlf.xi(lam_eff_filter/(1.+self.z))
        ltheta = 10.**(lNH-22) * ltheta_fact
        ltheta_2D = np.tile(ltheta, [len(lLlam_obs_grid), 1])

        #Before calculating the maximum host luminosity allowed, let's pre-compute a number of useful quantities. 
        # - Find the unscaled luminosity ratio. 
        Lnu_agn_unscaled = self.agn_sed.Lnu(lam_eff_filter/(1.+self.z))
        Lnu_hosts_unscaled = self.host_seds.Lnu(lam_eff_filter/(1.+self.z))
        x_unscaled_all = Lnu_hosts_unscaled/Lnu_agn_unscaled[0]
        # - Figure out the reddening factors for the AGN. 
        Aband_to_Alam = dict()
        klam_norm = self.agn_sed.klam(lam_eff_filter/(1.+self.z))
        for j, bp in enumerate(self.agn_sed.bps):
            lam_eff_band = bp.avgwave()
            bp_name = self.agn_sed.bp_names[j]
            Aband_to_Alam[bp_name] = (self.agn_sed.klam(lam_eff_band/(1+self.agn_sed.z)) / klam_norm)

        #Calculate the maximum host luminosity we can tolerate for each value of the reddening being considered for each host template. 
        Lh_La_max = np.zeros((lNH.shape[0],self.host_seds.likelihood.shape[0]))
        A_lams = 2.5*ltheta
        for k, A_lam in enumerate(A_lams):
            Lh_La_max[k] = Lhost_Lagn_max(self.agn_sed, self.host_seds, A_lam, Aband_to_Alam, self.sel_crit, x_unscaled_all)

        #Estimate some useful quantities.
        #Lfrac = 10**lLfrac
        nu_rest = (c/(lam_eff_filter/(1.+self.z))).to(u.Hz)
        L_nu    = (10**lLlam_obs_grid)/nu_rest * self.qlf.Lstar_units
        L_nu_2D = np.tile(L_nu, [len(lNH), 1]).T
        #L_bol_AGN = Lfrac*Lstar
        L_bol_AGN = 10**(lLbol) * self.qlf.Lstar_units
        L_bol_AGN_2D = np.tile(L_bol_AGN, [len(lNH), 1]).T

        #This is the array that will hold the combined probabilities. 
        P_2D = np.zeros(L_bol_AGN_2D.shape)

        #Iterate by template to fill out the P_2D array.
        for k_sed in range(len(self.host_seds.sps)):
            if self.host_seds.likelihood[k_sed] == 0.:
                continue
            Lh_La_max_2D = np.tile(Lh_La_max[:,k_sed], [len(lLlam_obs_grid),1])
            Lh_nu_max_2D = Lh_La_max_2D * L_nu_2D

            P_host = self.glf.P(Lh_nu_max_2D, hosts_sed=self.host_seds, k_hosts_sed=k_sed, L_AGN=L_bol_AGN_2D, lam_rest=lam_eff_filter/(1.+self.z))
            P_2D += self.host_seds.likelihood[k_sed] * P_host

        #For each NH, we will need to evaluate the unreddened QLF at a luminosity of lLfrac_lam_obs_grid + ltheta. So let's build it as a 2D array in which each row has the same lLfrac_lam_obs_grid value modified by the reddening correction (i.e., unreddened assuming different levels of obscuration).
        lLfrac_lam_sig_eval_2D = np.tile(lLlam_obs_grid, [len(lNH), 1]).T + ltheta_2D

        #Now, evaluate the f_NH function, following the S20 pubtools. Note: I think this actually wrong. f_NH should be evaluated at the intrinsic luminosity fraction of the reddening corrected luminosity. Here, we just assume that the same intrinsic lLfrac corresponds to the observed lLfrac_lam_obs_grid value for all NHs.
        lLfrac = lLbol - self.qlf.log_Lstar(self.z)
        lLfrac_eval_2D = np.tile(lLfrac, [len(lNH),1]).T
        log_NH_2D = np.tile(lNH, [len(lLlam_obs_grid), 1])
        Lstar = 10.**(self.qlf.log_Lstar(self.z))*self.qlf.Lstar_units
        Lstar_10 = (Lstar/(1e10*L_sun)).to(1.).value
        f_NH = self.qlf.fNH(log_NH_2D, lLfrac_eval_2D, Lstar_10, self.z)

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


