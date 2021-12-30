import numpy as np
import astropy.units as u 
import matplotlib.pyplot as plt

#from Phi_Obs_hostgal_MiLim import get_phi_lam_obs
from Phi_Obs_v2_MiLim import get_phi_lam_obs
from QLFs.Shen20 import QLF

qlf = QLF()

lL_frac_min = -3.0
lL_frac_max =  3.0

phi, dlLfrac = get_phi_lam_obs(1.0, qlf, lL_frac_min, lL_frac_max, 5000.*u.angstrom)

lLfracs = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac.value, dlLfrac.value)

plt.plot(lLfracs, phi)
plt.yscale('log')
plt.show()
