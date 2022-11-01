import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp2d
import astropy.units as u

from .agnLeddBased import AGNLeddBased

class Ananna22(AGNLeddBased):

    def __init__(self, **kargs):

        self.log_lam_e_star = -1.338
        self.log_xi_star = -3.64
        self.xi_star = 10**(self.log_xi_star)
        self.delta_1 = 0.38
        self.eps_lam = 2.260

        self.log_MBH_star = 7.88
        self.log_phi_star = -3.52
        self.phi_star = 10.**(self.log_phi_star)
        self.alpha = -1.576
        self.beta = 0.593

        #Bounds for integration. Exceed the limits from the BASS sample, but better that way for not having issues.
        log_MBH1 = 5.
        log_MBH2 = 10.
        dlog_MBH = 0.1
        log_MBH = np.arange(log_MBH1, log_MBH2+0.1*dlog_MBH, dlog_MBH)

        log_Lbol1 = 35.
        log_Lbol2 = 52.
        dlog_Lbol = 0.25
        log_Lbol = np.arange(log_Lbol1, log_Lbol2+0.1*dlog_Lbol, dlog_Lbol)

        #Now, pre-compute the integrals, so that when the code runs, it is just interpolating between the points needed. 
        precomp_P = np.zeros((len(log_Lbol), len(log_MBH)))
        for j, log_Lbol_use in enumerate(log_Lbol):
            norm = quad(self.func, log_MBH1, log_MBH2, args=(log_Lbol_use))[0]
            for i, log_MBH_max in enumerate(log_MBH):
                precomp_P[j,i] = quad(self.func, log_MBH1, log_MBH_max, args=(log_Lbol_use))[0]/norm
        
        #Finally, create the interpolation object.
        self.P_interp = interp2d(log_MBH, log_Lbol, precomp_P)

        return

    def P(self, Lh_nu_max, **kargs):

        #Get the maximum M_BH allowed given the maximum level of host galaxy permited. 
        lM_BH_max = self.lM_BH_max(Lh_nu_max, **kargs)

        #Now, compute the fraction of sources below that BH mass considering the Eddington ratio they would need to have to produce the observed luminosity L_AGN.
        log_Lagn = np.log10(kargs['L_AGN'].to(u.erg/u.s).value)

        #There is a severe problem for interp2d, in that no matter how one calls it, it will sort the input arguments and the result will be for the sorted arguments. So we need to call a sort in each loop, which makes the whole thing slower than it needs to be. NEED TO FIND A BETTER SOLUTION HERE! 
        Prob = np.zeros(Lh_nu_max.shape)
        for i in range(Lh_nu_max.shape[0]):
            idx = np.argsort(lM_BH_max[i,:])
            Paux = self.P_interp(lM_BH_max[i,:], log_Lagn[i,0])
            Prob[i,idx] = Paux
        return Prob

    def xi(self, log_lam_e):
        x = (log_lam_e-self.log_lam_e_star)
        return 10.**(-self.delta_1 * x) / (1 + 10.**(self.eps_lam * x))

    def phi(self, log_MBH):
        x = log_MBH - self.log_MBH_star
        return 10.**((self.alpha+1)*x) * np.exp(-10**(self.beta * x))

    def func(self, log_MBH, log_Lbol):
        log_lam_e = log_Lbol - log_MBH - 38.18
        return self.phi(log_MBH) * self.xi(log_lam_e)
