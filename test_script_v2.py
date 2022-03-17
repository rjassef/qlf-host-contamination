import numpy as np
import astropy.units as u 
from astropy.constants import c
import matplotlib.pyplot as plt

from Phi_Obs_hostgal_MiLim_v2 import get_phi_lam_obs
#from Phi_Obs_v2_MiLim import get_phi_lam_obs
from qlfhosts.QLFs.Shen20 import QLF

from qlfhosts.SEDs import A10_AGN, A10_hosts
from qlfhosts.AGN_Selection.R02_v2 import R02
from qlfhosts.GLFs.Willmer06 import Willmer06
from qlfhosts.GLFs.Uniform import Uniform
from qlfhosts.GLFs.Kollmeier06 import Kollmeier06

#Set the redshift at which we want to estimate the observed QLF. 
#z = 1.0
z = 0.5

#Set the observed wavelength at which to estimate the QLF.
lam_eff_filter = 5000.*u.angstrom

#Set the band names.
bp_names = ['sdssu', 'sdssg', 'sdssr']

#Set up the SED models to use. 
agn_sed = A10_AGN(z, bp_names=bp_names)
hosts_sed = A10_hosts(z, bp_names=bp_names)
hosts_sed.likelihood = np.array([0. ,1. ,0.])

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
phi, dlLfrac = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf)

#Estimate now without the selection applied considering reddening and host contamination.
glf2 = Uniform()
phi2, dlLfrac2 = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf2)

#Estimate now with the Eddington-ratio dependent host distribution.
#glf3 = Kollmeier06()
#phi3, dlLfrac3 = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf3)

#Plot the observed QLF.
lLfracs = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac.value, dlLfrac.value)
lLfracs2 = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac2.value, dlLfrac2.value)
#lLfracs3 = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac3.value, dlLfrac3.value)

Lstar = 10.**(qlf.log_Lstar(z)) * qlf.Lstar_units
nu_Lstarnu = qlf.L_at_lam(Lstar, lam_eff_filter/(1.+z))

nu_rest = c/(lam_eff_filter/(1.+z))
Lstar_nu = (nu_Lstarnu/nu_rest).to(u.erg/u.s/u.Hz).value

norm = np.log10(nu_Lstarnu.to(u.erg/u.s).value)
lLnu = lLfracs + norm
lLnu2 = lLfracs2 + norm
#lLnu3 = lLfracs3 + norm

lphi = np.log10(phi.value)
lphi2 = np.log10(phi2.value)
#lphi3 = np.log10(phi3.value)

#plt.plot(lLfracs, phi)
#plt.plot(lLfracs2, phi2)
#plt.yscale('log')
#plt.ylim([1e-7, 1e-4])

plt.plot(lLnu , lphi , label='Willmer06')
#plt.plot(lLnu2, lphi2, label='No host/selection')
#plt.plot(lLnu3, lphi3, label='Kollmeier06')
#plt.xlim([39.56, 51.03])
#plt.ylim([-17.3, -2.94])
plt.xlim([40., 46.])
plt.ylim([-7., -3.5])

plt.legend()
plt.title('Predicted AGN Luminosity Functions')
plt.xlabel(r'log Observed Luminosity in i-band ($\rm erg~\rm s^{-1})$')
plt.ylabel(r'log Space Density ($\rm dex^{-1}~\rm cMpc^{-3})$')

#plt.show()
plt.savefig("Predicted_AGN_LF_v2.png")
