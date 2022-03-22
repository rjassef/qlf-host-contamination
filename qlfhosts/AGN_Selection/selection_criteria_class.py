import numpy as np
import astropy.units as u
import sys 

class SelectionCriteria(object):

    def __init__(self):
        return

    #def is_selected(self,Lhost_Lagn, Alam, lam_rest, agn_sed, hosts_sed, k_hosts_sed, Lhost_Lagn_unscaled):
        
    def is_selected(self, Lhost_Lagn, Lhost_Lagn_unscaled, agn_mags, mag_diff):

        #Set the normalization for the SED ratios.
        ratio = Lhost_Lagn/Lhost_Lagn_unscaled

        #Clause case for when there is no host. 
        if ratio==0.:
            return self.is_agn(agn_mags)

        #Otherwise, let's move on and take the log of the ratio.
        lratio = np.log10(ratio)

        #Get the magnitudes.  Note that mag_diff is simply mag_host - mag_agn. The quantity zeta is a convenient quantity to separate extreme cases such that we do not overflow the calculations. 
        mag = dict()
        for bp_name in agn_mags:
            zeta = mag_diff[bp_name] - 2.5*lratio
            #First if clause is in the case where the AGN is much brighter than the host. In that case, the AGN will dominate the magnitudes. 
            if zeta > 20:
                mag[bp_name] = agn_mags[bp_name]
            #In this second clause, the host is much brighter than the AGN. In this case, the host dominate the magnitudes. 
            elif zeta < -20:
                mag[bp_name] = agn_mags[bp_name] + zeta
            else:    
                #mag[bp_name] = agn_mags[bp_name] - 2.5*np.log10(1.+ratio*10**(-0.4*mag_diff[bp_name]))
                mag[bp_name] = agn_mags[bp_name] - 2.5*np.log10(1.+10**(-0.4*zeta))

        #Apply the selection criterion. 
        return self.is_agn(mag)
