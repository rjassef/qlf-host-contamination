import numpy as np 
from SED_Model import lrt_model

class lrt_SED_model(object):

    def __init__(self, z, host_type, lam):

        #Save the input parameters. 
        self.z = z
        self.lam = lam

        #Save and renormalize the host_type if necessary. 
        self.host_type = host_type/np.sum(host_type)

        #Read the band zero poibts.
        self.jyzero = np.loadtxt("bandmag.dat", usecols=[2])

        #Create the object. 
        self.sed = lrt_model()
        self.sed.zspec = z
        self.sed.igm = 1.0
        self.sed.ebv = 0.0

        #Start all bolometric luminosities of the components at zero.
        self.sed.comp = np.zeros(4)

        #Because the LRT models work using the bolometric luminosities of the templates, we need to figure out how the bolometric luminosity ratio between the host and the AGN relates to the ratio at the specific wavelength. Note that we are interested in the ratio without AGN reddening. We compute here the ratio at the inpur wavelength for an SED with equal host and AGN luminosity.
        self.sed.comp[1:] = host_type
        self.L_at_lam_host = self.sed.L_at_lam(lam)
        self.sed.comp[1:] = 0
        self.sed.comp[0] = 1.0
        self.L_at_lam_agn = self.sed.L_at_lam(lam)

        #This is the conversion factor. If we take a bolometric ratio and we multiply it by this factor, we get the ratio at wavelength lambda.
        self.bol_ratio_to_lam_ratio_factor = self.L_at_lam_host/self.L_at_lam_agn

        return

    def get_mags(self, ebv, Lhost_Lagn_lam):

        #Set the reddening.
        self.sed.ebv = ebv

        #Set the host component noting that the agn component bolometric luminosity is 1.0 in these units. Since the user input is the ratio at wavelength lambda, we need to convert it to a bolometric luminosity ratio first. 
        Lhost_Lagn = Lhost_Lagn_lam/self.bol_ratio_to_lam_ratio_factor
        self.sed.comp[1:] = self.host_type * Lhost_Lagn

        #Get the model fluxes.
        self.sed.get_model_fluxes()

        #Calculate the magnitudes.
        self.mag = -2.5*np.log10(self.sed.jymod/self.jyzero)

        return

