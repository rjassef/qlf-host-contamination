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

        #Get the magnitudes.  
        mag = dict()
        for bp_name in agn_mags:
            mag[bp_name] = agn_mags[bp_name] - 2.5*np.log10(1.+ratio*10**(-0.4*mag_diff[bp_name]))

        #Apply the selection criterion. 
        return self.is_agn(mag)
