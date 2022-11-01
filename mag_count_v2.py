import numpy as np
import astropy.units as u 
from astropy.constants import c, L_sun
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from regex import P
from scipy.interpolate import interp1d

from qlfhosts.GLFs import Uniform, Willmer06, Kollmeier06, Ananna22

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

#Set the GLFs to use.
glfs = [
    Uniform,
    Willmer06,
    Kollmeier06,
    Ananna22,
]

#Set the observed magnitude boundaries. 
m_faint = 28.0
m_bright = 15.0
m_zp_jy = 3631.0*u.Jy

#Set the observed magnitude grid.
dmag = 0.5 * u.mag
m_grid = np.arange(m_bright, m_faint+0.1*dmag.value, dmag.value)
Ntot = np.zeros((len(glfs),m_grid.shape[0]))

#Iterate on redshift. 
for k, z in enumerate(zuse):

    for i, glf in enumerate(glfs):
    
        #Estimate the observed QLF.
        phi_obj = PhiObs(z, Galaxy_Luminosity_Distribution=glf)

        DL = DLs[k]
        lfact = np.log10(m_zp_jy * 4*np.pi*DL**2 * c/lam_eff_filter / phi_obj.qlf.Lstar_units)
        lLlam_obs_min = lfact - 0.4*m_faint
        lLlam_obs_max = lfact - 0.4*m_bright       

        phi, dlLlam = phi_obj.get_phi_lam_obs(lLlam_obs_min, lLlam_obs_max, lam_eff_filter)

        #Get the luminosity values.
        lLlam = np.arange(lLlam_obs_min, lLlam_obs_max+0.1*dlLlam.value, dlLlam.value)

        #Convert to observed magnitude. 
        nu_rest = (c*(1+z)/lam_eff_filter)
        Lnu = 10**lLlam * phi_obj.qlf.Lstar_units/nu_rest
        Fnu = Lnu*(1+z)/(4*np.pi*DL**2)
        mag = -2.5*np.log10(Fnu/m_zp_jy)

        #Get the comoving volume element.
        Vc = Vcs[k]

        #Interpolate in the histogram grid.
        phi_interp1 = interp1d(mag, phi.value, fill_value='extrapolate')

        #We have to multiple by -1 since mag/dex = -2.5 within astropy.units. There is probably a more elegant wat to do this, but this works. 
        Ntot[i] += (phi_interp1(m_grid)*phi.unit * Vc * dmag * -1).to(1).value


plt.xlim([15, 28])
plt.tick_params(top=True, right=True, grid_alpha=0.5)
plt.grid()
plt.plot(m_grid, Ntot[0], label='Without Selection Function')
plt.plot(m_grid, Ntot[1], label='With Selection Function but no AGN-host Relation')
plt.plot(m_grid, Ntot[2], label='With Selection Function and AGN-host Relation')
plt.plot(m_grid, Ntot[3], label='With Selection Function and AGN-host Relation 2')
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
np.savetxt("Quasar_mag_counts_g_v2.dat", np.array([m_grid, Ntot[0], Ntot[1], Ntot[2]]))
