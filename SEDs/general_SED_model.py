import numpy as np 
import astropy.units as u
import os
from astropy.utils.data import get_pkg_data_filename
from synphot import SpectralElement, units, Observation
from synphot.models import Empirical1D

class general_SED_model(object):

    def __init__(self, sed_template, z, bp_names=['sdssu'], bp_folder=None):
        '''
        This is a general code to handle SEDs and calculate their magnitudes and colors. It relies on the synphot_refract package.

        '''

        #Load the bandpasses.
        bp = list()
        for bp_name in bp_names:
            if bp_folder is None:
                filename = get_pkg_data_filename(os.path.join('bandpasses', bp_name+".filter"))
            else:
                filename = os.path.join(bp_folder, bp_name)
            bp.append(SpectralElement.from_file(filename))

        #Load the SED template.
        sp = None
        sp.z = z

        #Create the observations of the spectrum in all the bands.
        obs = Observation(sp, bp, binset=bp.binset)

        return