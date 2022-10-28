import numpy as np
import astropy.units as u 
from astropy.constants import c, L_sun
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from regex import P
from scipy.interpolate import interp1d

#from Phi_Obs_hostgal_MiLim import get_phi_lam_obs
#from phi_obs_general import get_phi_lam_obs
from qlfhosts.QLFs import S20_QLF as QLF

#from qlfhosts.SEDs import R06_AGN, A10_hosts
#from qlfhosts.AGN_Selection import R02
from qlfhosts.GLFs import Uniform, Willmer06, Kollmeier06

from qlfhosts.util.phiObs import PhiObs

#Set the redshift range and the number of logarithmically separated steps within which to count the number of AGN. 
zmin = 0.3
zmax = 6.7
nz = 50
zs = np.logspace(np.log10(zmin), np.log10(zmax), nz)
dzs = zs[1:]-zs[:-1]

#Pre calculate the luminosity distances. Evaluate at the mid-point of the redshift range.
zuse = zs[:-1] + 0.5*dzs
DLs = cosmo.luminosity_distance(zuse)
Vcs = cosmo.comoving_volume(zs[1:])-cosmo.comoving_volume(zs[:-1])

#Set the observed wavelength at which to estimate the QLF.
lam_eff_filter = 4750.*u.angstrom

#Set the observed magnitude boundaries. 
m_faint = 28.0
m_bright = 15.0
m_zp_jy = 3631.0*u.Jy

#Set the observed magnitude grid.
dmag = 0.5 * u.mag
m_grid = np.arange(m_bright, m_faint+0.1*dmag.value, dmag.value)
Ntot1 = np.zeros(m_grid.shape)
Ntot2 = np.zeros(m_grid.shape)
Ntot3 = np.zeros(m_grid.shape)

#Well need to load the QLF for plotting. But should get this sent into the PhiObs code itself.
qlf = QLF()

#Iterate on redshift. 
for k, z in enumerate(zuse):

    #Transform the magnitude limits to boundaries. 
    #DL = cosmo.luminosity_distance(z)
    DL = DLs[k]
    lfact = np.log10(m_zp_jy * 4*np.pi*DL**2 * c/lam_eff_filter / qlf.Lstar_units)
    lLlam_obs_min = lfact - 0.4*m_faint
    lLlam_obs_max = lfact - 0.4*m_bright

    #Estimate the observed QLF.
    phi1_obj = PhiObs(z, Galaxy_Luminosity_Distribution=Uniform)
    phi1, dlLlam1 = phi1_obj.get_phi_lam_obs(lLlam_obs_min, lLlam_obs_max, lam_eff_filter)

    phi2_obj = PhiObs(z, Galaxy_Luminosity_Distribution=Willmer06)
    phi2, dlLlam2 = phi2_obj.get_phi_lam_obs(lLlam_obs_min, lLlam_obs_max, lam_eff_filter)

    phi3_obj = PhiObs(z, Galaxy_Luminosity_Distribution=Kollmeier06)
    phi3, dlLlam3 = phi3_obj.get_phi_lam_obs(lLlam_obs_min, lLlam_obs_max, lam_eff_filter)

    #Get the luminosity values.
    lLlam1 = np.arange(lLlam_obs_min, lLlam_obs_max+0.1*dlLlam1.value, dlLlam1.value)
    lLlam2 = np.arange(lLlam_obs_min, lLlam_obs_max+0.1*dlLlam2.value, dlLlam2.value)
    lLlam3 = np.arange(lLlam_obs_min, lLlam_obs_max+0.1*dlLlam3.value, dlLlam3.value)

    #Convert to observed magnitude. 
    nu_rest = (c*(1+z)/lam_eff_filter)
    Lnu1 = 10**lLlam1 * qlf.Lstar_units/nu_rest
    Fnu1 = Lnu1*(1+z)/(4*np.pi*DL**2)
    mag1 = -2.5*np.log10(Fnu1/m_zp_jy)
    Lnu2 = 10**lLlam2 * qlf.Lstar_units/nu_rest
    Fnu2 = Lnu2*(1+z)/(4*np.pi*DL**2)
    mag2 = -2.5*np.log10(Fnu2/m_zp_jy)
    Lnu3 = 10**lLlam3 * qlf.Lstar_units/nu_rest
    Fnu3 = Lnu3*(1+z)/(4*np.pi*DL**2)
    mag3 = -2.5*np.log10(Fnu3/m_zp_jy)

    #Get the comoving volume element.
    #Vc = cosmo.comoving_volume(zs[k+1])-cosmo.comoving_volume(zs[k])
    Vc = Vcs[k]

    #Interpolate in the histogram grid.
    phi_interp1 = interp1d(mag1, phi1.value, fill_value='extrapolate')
    phi_interp2 = interp1d(mag2, phi2.value, fill_value='extrapolate')
    phi_interp3 = interp1d(mag3, phi3.value, fill_value='extrapolate')

    #We have to multiple by -1 since mag/dex = -2.5 within astropy.units. There is probably a more elegant wat to do this, but this works. 
    Ntot1 += (phi_interp1(m_grid)*phi1.unit * Vc * dmag * -1).to(1).value
    Ntot2 += (phi_interp2(m_grid)*phi2.unit * Vc * dmag * -1).to(1).value
    Ntot3 += (phi_interp3(m_grid)*phi3.unit * Vc * dmag * -1).to(1).value

#plt.plot(m_grid, Ntot1, label='Uniform')
#plt.plot(m_grid, Ntot2, label='Willmer06')
#plt.plot(m_grid, Ntot3, label='Kollmeier06')
plt.xlim([15, 28])
plt.tick_params(top=True, right=True, grid_alpha=0.5)
plt.grid()
plt.plot(m_grid, Ntot1, label='Without Selection Function')
plt.plot(m_grid, Ntot2, label='With Selection Function but no AGN-host Relation')
plt.plot(m_grid, Ntot3, label='With Selection Function and AGN-host Relation')
plt.yscale('log')
plt.legend()
plt.xlabel("g-band magnitude")
plt.ylabel("Number counts")

#Draw the nominal 5 sigma depth of the WFD. 
plt.axvline(27.4, color='black', linestyle='dashed')
yloc = 10**(np.mean(np.log10(plt.ylim()))-1.0)
plt.text(27.0, yloc, r'WFD $5\sigma$ depth', rotation='vertical')

plt.savefig("Quasar_mag_counts_g_v2.png", dpi=200)

#Save the results.
np.savetxt("Quasar_mag_counts_g_v2.dat", np.array([m_grid, Ntot1, Ntot2, Ntot3]))
