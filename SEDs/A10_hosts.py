from http.client import LineTooLong
import numpy as np 
import astropy.units as u
import os
from astropy.utils.data import get_pkg_data_filename
from synphot import SpectralElement, units, Observation
from synphot.models import Empirical1D
from astropy.constants import c, L_sun
import astropy.units as u
from synphot import SourceSpectrum

from .general_SED_model import general_SED_model


class A10_hosts(general_SED_model):

    def __init__(self, z, bp_names=['sdssu'], bp_folder=None, cosmo=None, likelihood=None):

        #Read the A10 SEDs. 
        fname = get_pkg_data_filename(os.path.join('SED_templates', 'A10_SEDs.dat'))
        data = np.loadtxt(fname, skiprows=1)

        #Get the wavelengths from the frequency. 
        lam = data[:,0] * u.micron
        nu = (c/lam).to(u.Hz)

        #Get the luminosity distance.
        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        DL = cosmo.luminosity_distance(z)

        #Go through each of the three host SEDs.
        sps = list()
        for i in range(3):

             #The SED is in unit of Fnu at 10pc for a bolometric luminosity of 10^10 L_sun, although this is not strictly correct. I divide by 10^10 to get more reasonable magnitudes, although the code itself is independent of the exact normalization at this point. 
            Lnu = data[:,i+3] * u.erg/u.s/u.cm**2/u.Hz * 4*np.pi*(10*u.pc)**2 / 1e10
            fnu_obs = (Lnu*(1+z)/(4.*np.pi*DL**2)).to(u.erg/u.s/u.cm**2/u.Hz)

            #Create the synphot spectrum object.
            sp = SourceSpectrum(Empirical1D, points=lam, lookup_table=fnu_obs)
            sp.z = z 

            #Add to the list.
            sps.append(sp)

        #Set the likelihood. If none is provided, set it to be proportional to the number of host templates.
        if likelihood is None:
            self.likelihood = np.ones(len(sps))/len(sps)
        else:
            self.likelihood = np.array(likelihood)

        #Initialize the general SED model. 
        super().__init__(sps, z, bp_names, bp_folder)

        return 
