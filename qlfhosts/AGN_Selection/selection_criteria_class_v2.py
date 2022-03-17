import numpy as np
import astropy.units as u
import sys 

class SelectionCriteria(object):

    def __init__(self):
        return

    def is_selected(self,Lhost_Lagn, Alam, lam_rest, agn_sed, hosts_sed, k_hosts_sed, Lhost_Lagn_unscaled):
        
        #Set the normalization for the SED ratios.
        ratio = Lhost_Lagn/Lhost_Lagn_unscaled[k_hosts_sed]

        #Figure out the reddening factors for the AGN. 
        Aband = np.zeros(len(agn_sed.bps))
        klam_norm = agn_sed.klam(lam_rest)
        for j, bp in enumerate(agn_sed.bps):
            lam_eff_band = bp.avgwave()
            Aband[j] = (agn_sed.klam(lam_eff_band/(1+agn_sed.z)) / klam_norm) * Alam

        #Get the AGN magnitudes.
        agn_mags = dict()
        for j, bp_name in enumerate(agn_sed.bp_names):
            agn_mags[bp_name] = agn_sed.mag_unscaled[0][j] + Aband[j]

        #Get the host magnitudes. 
        host_mags = dict()
        for j, bp_name in enumerate(hosts_sed.bp_names):
            host_mags[bp_name] = hosts_sed.mag_unscaled[k_hosts_sed][j]

        #Combine the magnitudes.
        try:
            mag = dict()
            for bp_name in agn_mags:
                mag[bp_name] = -2.5*np.log10(10**(-0.4*agn_mags[bp_name]) + ratio * 10**(-0.4*host_mags[bp_name]))
        except KeyError:
            print("Need to use the same bands in AGN and hosts SEDs.")
            sys.exit()
   
        #Apply the selection criterion. 
        return self.is_agn(mag)
