import numpy as np
import astropy.units as u 
from astropy.constants import c, L_sun
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import interp1d

#from Phi_Obs_hostgal_MiLim import get_phi_lam_obs
from phi_obs_general import get_phi_lam_obs
from qlfhosts.QLFs import S20_QLF as QLF

from qlfhosts.SEDs import R06_AGN, A10_hosts
from qlfhosts.AGN_Selection import R02
from qlfhosts.GLFs import Uniform

#Set the redshift at which we want to estimate the observed QLF. 
#z = 1.0
z = 0.5

#Set the observed wavelength at which to estimate the QLF.
#lam_eff_filter = 7500.*u.angstrom
lam_eff_filter = 5000.*u.angstrom

#Set the observed magnitude boundaries. 
m_faint = 28.0
m_bright = 15.0
m_zp_jy = 3631.0*u.Jy

#Set the observed magnitude grid.
dmag = 0.5
m_grid = np.arange(m_bright, m_faint+0.1*dmag, dmag)
Ntot = np.zeros(m_grid.shape)
#print(m_grid)
#exit()

#Set the band names.
bp_names = ['sdssu', 'sdssg', 'sdssr']

#Set up the intrinsic qlf to use. 
qlf = QLF()

#Set up the AGN selection criteria.
sel_crit = R02()

#Set the redshift grid.
zs = np.logspace(-2, -1, 10)
dzs = zs[1:]-zs[:-1]
#zs = [0.5]

#Iterate on redshift. 
for k, dz in enumerate(dzs):

    z = zs[k]+0.5*dz

    #Set up the SED models to use. 
    agn_sed = R06_AGN(z, bp_names=bp_names, cosmo=cosmo)
    hosts_sed = A10_hosts(z, bp_names=bp_names, cosmo=cosmo)
    hosts_sed.likelihood = np.array([0. ,1. ,0.])

    #Set up the GLF to use.
    glf = Uniform()

    #Transform the magnitude limits to boundaries. 
    DL = cosmo.luminosity_distance(z)
    lfact = np.log10(m_zp_jy * 4*np.pi*DL**2 * c/lam_eff_filter / qlf.Lstar_units)
    lLlam_obs_min = lfact - 0.4*m_faint
    lLlam_obs_max = lfact - 0.4*m_bright

    #Estimate the observed QLF.
    phi, dlLlam = get_phi_lam_obs(z, qlf, lLlam_obs_min, lLlam_obs_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf)

    #print(phi.unit)
    #exit()

    #Get the luminosity values.
    lLlam = np.arange(lLlam_obs_min, lLlam_obs_max+0.1*dlLlam.value, dlLlam.value)

    #Convert to observed magnitude. 
    nu_rest = (c*(1+z)/lam_eff_filter)
    Lnu = 10**lLlam * qlf.Lstar_units/nu_rest
    Fnu = Lnu*(1+z)/(4*np.pi*DL**2)
    mag = -2.5*np.log10(Fnu/m_zp_jy)

    #Get the comoving volume element.
    Vc = cosmo.comoving_volume(zs[k+1])-cosmo.comoving_volume(zs[k])

    #Interpolate in the histogram grid.
    phi_interp = interp1d(mag, phi.value, fill_value='extrapolate')
    Ntot += phi_interp(m_grid)*phi.unit * Vc * dmag

    #plt.plot(lLlam+np.log10(L_sun.to(u.erg/u.s).value), np.log10(phi.value))
    #plt.plot(mag, phi*Vc)
    plt.plot(m_grid, Ntot)
    plt.yscale('log')
    #plt.ylim([10**(-7.), 10**(-3.5)])
    #plt.ylim([])
    plt.show()

exit()

#Estimate now without the selection applied considering reddening and host contamination.
glf2 = Uniform()
phi2, dlLfrac2 = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf2)

#Estimate now with the Eddington-ratio dependent host distribution.
glf3 = Kollmeier06()
phi3, dlLfrac3 = get_phi_lam_obs(z, qlf, lL_frac_min, lL_frac_max, lam_eff_filter , agn_sed, hosts_sed, sel_crit, glf3)

#Plot the observed QLF.
lLfracs = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac.value, dlLfrac.value)
lLfracs2 = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac2.value, dlLfrac2.value)
lLfracs3 = np.arange(lL_frac_min, lL_frac_max+0.1*dlLfrac3.value, dlLfrac3.value)

Lstar = 10.**(qlf.log_Lstar(z)) * qlf.Lstar_units
nu_Lstarnu = qlf.L_at_lam(Lstar, lam_eff_filter/(1.+z))

nu_rest = c/(lam_eff_filter/(1.+z))
Lstar_nu = (nu_Lstarnu/nu_rest).to(u.erg/u.s/u.Hz).value

norm = np.log10(nu_Lstarnu.to(u.erg/u.s).value)
lLnu = lLfracs + norm
lLnu2 = lLfracs2 + norm
lLnu3 = lLfracs3 + norm

lphi = np.log10(phi.value)
lphi2 = np.log10(phi2.value)
lphi3 = np.log10(phi3.value)

#plt.plot(lLfracs, phi)
#plt.plot(lLfracs2, phi2)
#plt.yscale('log')
#plt.ylim([1e-7, 1e-4])

plt.plot(lLnu , lphi , label='Willmer06')
plt.plot(lLnu2, lphi2, label='No host/selection')
plt.plot(lLnu3, lphi3, label='Kollmeier06')
#plt.xlim([39.56, 51.03])
#plt.ylim([-17.3, -2.94])
plt.xlim([40., 46.])
plt.ylim([-7., -3.5])

plt.legend()
plt.title('Predicted AGN Luminosity Functions')
plt.xlabel(r'log Observed Luminosity in i-band ($\rm erg~\rm s^{-1})$')
plt.ylabel(r'log Space Density ($\rm dex^{-1}~\rm cMpc^{-3})$')

#plt.show()
plt.savefig("Predicted_AGN_LF.R06.png")
