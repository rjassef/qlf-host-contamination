import numpy as np
import astropy.units as u 
import matplotlib.pyplot as plt
from astropy.table import Table

from qlfhosts.GLFs import Uniform, Willmer06, Kollmeier06, Ananna22

from qlfhosts.util.magCount import MagCount

#Set the GLFs to use.
glfs = [
    Uniform,
    Willmer06,
    Kollmeier06,
    Ananna22,
]

#Set the i-band wavelength. 
lam_eff_filter = 7501.62*u.AA

#Get the counts and save them. 
output = Table()
for k,glf in enumerate(glfs):
    Nagn_obj = MagCount(Galaxy_Luminosity_Distribution=glf)
    Ntot_aux, m_grid_aux = Nagn_obj.calc(lam_eff_filter=lam_eff_filter, dmag=0.1*u.mag)

    if k==0:
        output['mag'] = m_grid_aux
    model_name = str(glf.__name__)
    output[model_name]= Ntot_aux

#Save them. 
output.write("iband_mag_counts.dat",format='ascii')