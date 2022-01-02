import numpy as np
from mpmath import gammainc
from scipy.interpolate import interp1d 

#Note: We cannot use the scipy.special gammainc, as it is not defined for negative arguments.

class SchechterFunction(object):

    def __init__(self, alpha, Lstar):

        #Save the input values.
        self.alpha = alpha
        self.Lstar = Lstar

        #When -2 < alpha < -1 , as often found, the integral of the GLF formally diverges or becomes negative. Instead, we only count galaxies brighter than self.Lhost_to_Lstar_min times Lstar.
        self.Lhost_to_Lstar_min = 1e-3
        self.norm = np.float(gammainc(self.alpha+1, self.Lhost_to_Lstar_min))
        return

    def P(self, Lhost_max):

        #Divide by the GLF Lstar.
        Lhost_to_Lstar_max = (Lhost_max/self.Lstar).to(1).value

        #We will have to create an interpolation object to evaluate the integral of the GLF properly. 
        x_interp_max = np.max(Lhost_to_Lstar_max)
        x_interp_min = self.Lhost_to_Lstar_min
        x = np.logspace(np.log10(x_interp_min), np.log10(x_interp_max), 100)
        y = np.zeros(x.shape)
        for k, ix in enumerate(x):
            y[k] = np.float(gammainc(self.alpha+1, self.Lhost_to_Lstar_min, ix)) / self.norm
        Pfunc = interp1d(x, y, fill_value='extrapolate')

        #Interpolate for values above the integration minimum. For the rest, return 0.
        Prob = np.where(Lhost_to_Lstar_max>self.Lhost_to_Lstar_min, Pfunc(Lhost_to_Lstar_max), 0.)
        return Prob

        # Prob = np.zeros(Lhost_to_Lstar_max.shape)
        # for k in np.argwhere(Lhost_to_Lstar_max>self.Lhost_to_Lstar_min):
        #     Prob[tuple(k)] = np.float(gammainc(self.alpha+1, self.Lhost_to_Lstar_min, Lhost_to_Lstar_max[tuple(k)])) / self.norm
        #     if Prob[tuple(k)]>0.8:
        #         print(Prob[tuple(k)], Pfunc(Lhost_to_Lstar_max[tuple(k)]))
        #         input()

        # return Prob
