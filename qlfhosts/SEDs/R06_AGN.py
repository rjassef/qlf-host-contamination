import numpy as np 
import astropy.units as u
import os
from astropy.utils.data import get_pkg_data_filename
from synphot import SpectralElement, units, Observation
from synphot.models import Empirical1D
from astropy.constants import c
import astropy.units as u
from astropy.table import Table
import warnings
from synphot import SourceSpectrum

from .general_SED_model import general_SED_model


class R06_AGN(general_SED_model):

    def __init__(self, z, bp_names=['sdssu'], bp_folder=None, cosmo=None):

        #Read the R06 mean AGN SED.
        fname = get_pkg_data_filename(os.path.join('SED_templates', 'Richards_06.dat'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',u.UnitsWarning) 
            R06_SED_tab = Table.read(fname, format='ascii.cds')

        #Get the wavelengths from the frequency. 
        nu = 10**(R06_SED_tab['LogF'].data)*u.Hz
        lam = (c/nu).to(u.micron)

        #Get the luminosity distance.
        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        DL = cosmo.luminosity_distance(z)

        #The mean SED is in units of log(nu Lnu).
        Lnu = 10**(R06_SED_tab['All'].data)* u.erg/u.s/nu
        fnu_obs = (Lnu*(1+z)/(4.*np.pi*DL**2)).to(u.erg/u.s/u.cm**2/u.Hz)

        #Create the synphot spectrum object.
        sp = SourceSpectrum(Empirical1D, points=lam, lookup_table=fnu_obs)
        sp.z = z 

        #Initialize the general SED model. 
        super().__init__([sp], z, bp_names, bp_folder)

        return 

        