import numpy as np
import astropy.units as u 
from astropy.constants import c, L_sun
import matplotlib.pyplot as plt
from regex import P
from scipy.interpolate import interp1d

import multiprocessing as mp
from functools import partial


from qlfhosts.util.phiObs import PhiObs


class MagCount(object):

    def __init__(self, QLF=None, Selection=None, AGN_SED=None, Host_SEDs=None, Host_SED_likelihood=None, Galaxy_Luminosity_Distribution=None, cosmo=None, Mi_lim=None):

        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo
        self.cosmo = cosmo

        self.PhiObs_args = locals()
        del(self.PhiObs_args['self'])

        return


    def calc(self, zmin=0.3, zmax=6.7, nz=50, m_grid=None, m_faint = 28.0, m_bright = 15.0, m_zp_jy = 3631.0*u.Jy, dmag=0.5*u.mag, Ncpu=None, lNH_min=20., lNH_max=26., band='LSSTg', lam_eff_filter=None):

        #Set the effective wavelength of the observations.
        if band is not None:
            from .lam_eff import lam_eff
            self.lam_eff_filter = lam_eff[band]
        else:
            self.lam_eff_filter = lam_eff_filter

        #Set the redshift range and the number of logarithmically separated steps within which to count the number of AGN. 
        zs = np.logspace(np.log10(zmin), np.log10(zmax), nz)

        #Pre calculate the luminosity distances. Evaluate at the mid-point of the redshift range.
        dzs = zs[1:]-zs[:-1]
        zuse = zs[:-1] + 0.5*dzs
        DLs = self.cosmo.luminosity_distance(zuse)
        Vcs = self.cosmo.comoving_volume(zs[1:])-self.cosmo.comoving_volume(zs[:-1])

        #Set the observed magnitude grid.
        if m_grid is None:
            m_grid = np.arange(m_bright, m_faint+0.1*dmag.value, dmag.value)
        else:
            m_bright = None
            m_faint = None
            dmag = None

        if Ncpu is None:
            Ncpu = mp.cpu_count()
        Pool = mp.Pool(Ncpu)

        k_all = np.arange(0,len(zuse),1)
        k_split = np.array_split(k_all,Ncpu)
        func = partial(self.integrate_mag_count, m_grid, zuse, DLs, Vcs, m_zp_jy, lNH_min, lNH_max)
        Output = Pool.map(func, k_split)

        Ntot = np.sum(Output, axis=0)

        #Close the pool. This is important because otherwise resources remain taken and can run into a "Too many files open" error.
        Pool.close()

        return Ntot, m_grid


    def integrate_mag_count(self, m_grid, zuse, DLs, Vcs, m_zp_jy, lNH_min, lNH_max, kuse):

        #The magnitude grid contains the minimum and maximum of each magnitude bin. But we want to evaluate at the mid point of each bin and then multiply by the width of bin. 
        m_grid_eval = 0.5*(m_grid[1:]+m_grid[:-1])
        dmag_grid = (m_grid[1:]-m_grid[:-1])*u.mag
        m_faint = np.max(m_grid)
        m_bright = np.min(m_grid)

        #The output array.
        Ntot = np.zeros((m_grid_eval.shape[0]))

        #Iterate on redshift. 
        for k in kuse:

            #Set the redshift to use.
            z = zuse[k]
         
            #Create the luminosity function object with the specific GLF to simulate host contamination. 
            phi_obj = PhiObs(z, **self.PhiObs_args)

            #Here we set the integration limits. The luminosity function is returned between pairs of lambda L_lambda luminosity values at the rest-frame lambda. The factor lfact just converts between magnitudes and lambda L_lambda.
            DL = DLs[k]
            lfact = np.log10(m_zp_jy * 4*np.pi*DL**2 * c/self.lam_eff_filter / phi_obj.qlf.Lstar_units)
            lLlam_obs_min = lfact - 0.4*m_faint
            lLlam_obs_max = lfact - 0.4*m_bright       

            #Estimate the luminosity function. 
            phi, dlLlam = phi_obj.get_phi_lam_obs(lLlam_obs_min, lLlam_obs_max, self.lam_eff_filter, lNH_min=lNH_min, lNH_max=lNH_max)

            #Get the corresponding observed magnitudes for the natural grid of the QLF. 
            dmag = 2.5*dlLlam.value
            mag = np.arange(m_bright, m_faint+0.1*dmag, dmag)
            mag = mag[::-1]

            #Get the comoving volume element.
            Vc = Vcs[k]

            #Interpolate in the histogram grid.
            phi_interp1 = interp1d(mag, phi.value, fill_value='extrapolate')

            #We have to multiple by -1 since mag/dex = -2.5 within astropy.units. There is probably a more elegant wat to do this, but this works. 
            Ntot += (phi_interp1(m_grid_eval)*phi.unit * Vc * dmag_grid * -1).to(1).value

        return Ntot


