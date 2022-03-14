import numpy as np 
import astropy.units as u
import os
from astropy.utils.data import get_pkg_data_filename
from synphot import SpectralElement, units, Observation
from synphot.models import Empirical1D

class general_SED_model(object):

    def __init__(self, sps, z, bp_names=['sdssu'], bp_folder=None, cosmo=None):
        '''
        This is a general code to handle SEDs and calculate their magnitudes and colors. It relies on the synphot_refract package.

        '''

        #Save the input redshift and the bandpass names.
        self.z = z
        self.bp_names = bp_names

        #Load the bandpasses.
        self.bps = list()
        for bp_name in bp_names:
            if bp_folder is None:
                filename = get_pkg_data_filename(os.path.join('bandpasses', bp_name+".filter"))
            else:
                filename = os.path.join(bp_folder, bp_name)
            self.bps.append(SpectralElement.from_file(filename))

        #Redshift the SED templates.
        self.sps = sps
        for sp in self.sps:
            if sp.z==0:
                sp.z = z

        #Create the observations of each spectrum in all the bands.
        self.obs = list()
        for sp in self.sps:
            self.obs.append([])
            for bp in self.bps:
                self.obs[-1].append(Observation(sp, bp))

        #Precompute the apparent magnitudes
        self.mag_unscaled = np.zeros((len(self.sps),len(self.bps)))
        for i,sp_obs in enumerate(self.obs):
            for j,band_obs in enumerate(sp_obs):
                self.mag_unscaled[i,j] = band_obs.effstim(flux_unit='ABmag').value

        #Load the luminosity distance. It will be useful.
        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        self.DL = cosmo.luminosity_distance(z)

        return

    #Return the luminosity density of the input SED at the observed wavelength lam. 
    def Lnu(self, lam_rest):
        Lnu = np.zeros(len(self.sps))*u.erg/u.s/u.Hz
        for k,sp in enumerate(self.sps):
            fnu_obs = sp(lam_rest*(1+self.z), flux_unit='Fnu')
            Lnu[k] = (fnu_obs*4*np.pi*self.DL**2)/(1+self.z)
        return Lnu

    #Compute the apparent magnitudes for a given luminosity density.
    def mag(self, Lnu_norm, lam_rest):

        #Get the luminosity densities of the sed models.
        lam_obs = lam_rest*(1+self.z)
        Lnu_input = self.Lnu(lam_obs)

        #Now, calculate the magnitudes 
        norm = np.tile(Lnu_norm/Lnu_input, (len(self.bps),1)).T
        mag = self.mag_unscaled-2.5*np.log10(norm)
        return mag

