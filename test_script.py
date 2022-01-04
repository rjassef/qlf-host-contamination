import numpy as np
import astropy.units as u 
from astropy.constants import c
import matplotlib.pyplot as plt

from Phi_Obs_hostgal_MiLim import get_phi_lam_obs
#from Phi_Obs_v2_MiLim import get_phi_lam_obs
from QLFs.Shen20 import QLF

from SEDs.lrt_SED_model import lrt_SED_model
from AGN_Selection.R02 import R02
from GLFs.Willmer06 import Willmer06
from GLFs.Uniform import Uniform

#Set the redshift at which we want to estimate the observed QLF. 
#z = 1.0
z = 0.5

#Set the observed wavelength at which to estimate the QLF.
lam_eff_filter = 5000.*u.angstrom

#Set up the SED models to use. 
host_type = np.array([0., 1., 0.])
sed_model = lrt_SED_model(z, host_type, lam_eff_filter)

#Set up the intrinsic qlf to use. 
qlf = QLF()

#Set up the AGN selection criteria.
sel_crit = R02()

#Set up the GLF to use.
glf = Willmer06(z)

#Set up the observed luminosity range (in units of L*) over which we want to estimate the observed QLF. 
lL_frac_min = -5.0
lL_frac_max =  6.0

#Estimate the observed QLF.
phi, dlLfrac = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , sed_model, sel_crit, glf)

#Estimate now with now selection applied considering reddening and host contamination.
glf2 = Uniform()
phi2, dlLfrac2 = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , sed_model, sel_crit, glf2)

#Plot the observed QLF.
lLfracs = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac.value, dlLfrac.value)
lLfracs2 = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac2.value, dlLfrac2.value)

Lstar = 10.**(qlf.log_Lstar(z)) * qlf.Lstar_units
nu_Lstarnu = qlf.L_at_lam(Lstar, lam_eff_filter/(1.+z))

nu_rest = c/(lam_eff_filter/(1.+z))
Lstar_nu = (nu_Lstarnu/nu_rest).to(u.erg/u.s/u.Hz).value

norm = np.log10(nu_Lstarnu.to(u.erg/u.s).value)
lLnu = lLfracs + norm
lLnu2 = lLfracs2 + norm

lphi = np.log10(phi.value)
lphi2 = np.log10(phi2.value)

#plt.plot(lLfracs, phi)
#plt.plot(lLfracs2, phi2)
#plt.yscale('log')
#plt.ylim([1e-7, 1e-4])

plt.plot(lLnu, lphi)
plt.plot(lLnu2, lphi2)
#plt.xlim([39.56, 51.03])
#plt.ylim([-17.3, -2.94])
plt.xlim([40., 46.])
plt.ylim([-7., -3.5])

plt.show()
