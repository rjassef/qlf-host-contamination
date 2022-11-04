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

    def __init__(self, QLF=None, Selection=None, AGN_SED=None, Host_SEDs=None, Host_SED_likelihood=None, Galaxy_Luminosity_Distribution=None, cosmo=None):

        if cosmo is None:
            from astropy.cosmology import Planck15 as cosmo

        self.PhiObs_args = locals()
        del(self.PhiObs_args['self'])

        self.cosmo = cosmo

        return


    def calc(self, zmin=0.3, zmax=6.7, nz=50,lam_eff_filter=4750.*u.angstrom, m_faint = 28.0, m_bright = 15.0, m_zp_jy = 3631.0*u.Jy, dmag=0.5*u.mag, Ncpu=None):

        #Set the redshift range and the number of logarithmically separated steps within which to count the number of AGN. 
        zs = np.logspace(np.log10(zmin), np.log10(zmax), nz)

        #Pre calculate the luminosity distances. Evaluate at the mid-point of the redshift range.
        dzs = zs[1:]-zs[:-1]
        zuse = zs[:-1] + 0.5*dzs
        DLs = self.cosmo.luminosity_distance(zuse)
        Vcs = self.cosmo.comoving_volume(zs[1:])-self.cosmo.comoving_volume(zs[:-1])

        #Set the observed magnitude grid.
        m_grid = np.arange(m_bright, m_faint+0.1*dmag.value, dmag.value)

        if Ncpu is None:
            Ncpu = mp.cpu_count()
        Pool = mp.Pool(Ncpu)

        k_all = np.arange(0,len(zuse),1)
        k_split = np.array_split(k_all,Ncpu)
        func = partial(self.get_mag_count, m_grid, zuse, DLs, Vcs, m_faint, m_bright, dmag, m_zp_jy, lam_eff_filter)
        Output = Pool.map(func, k_split)

        Ntot = np.sum(Output, axis=0)

        return Ntot, m_grid


    def get_mag_count(self, m_grid, zuse, DLs, Vcs, m_faint, m_bright, dmag, m_zp_jy, lam_eff_filter, kuse):

        Ntot = np.zeros((m_grid.shape[0]))

        #Iterate on redshift. 
        for k in kuse:

            #Set the redshift to use.
            z = zuse[k]
         
            #Estimate the observed QLF.
            phi_obj = PhiObs(z, **self.PhiObs_args)

            #Set the luminosity integration limits.
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
            Ntot += (phi_interp1(m_grid)*phi.unit * Vc * dmag * -1).to(1).value

        return Ntot


