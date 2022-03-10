import numpy as np 
import astropy.units as u
import os
from astropy.utils.data import get_pkg_data_filename
from synphot import SpectralElement

class general_SED_model(object):

    def __init__(self, bp_names=['sdssu'], bp_folder=None):
        '''
        This is a general code to handle SEDs and calculate their magnitudes and colors. It relies on the synphot_refract package.

        '''

        #Load the u-band as a test
        for bp_name in bp_names:
            if bp_folder is None:
                filename = get_pkg_data_filename(os.path.join('bandpasses', bp_names))
            else:
                filename = os.path.join(bp_folder, bp_name)


        return