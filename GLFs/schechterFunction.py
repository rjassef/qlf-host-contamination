import numpy as np
from mpmath import gammainc

#Note: We cannot use the scipy.special gammainc, as it is not defined for negative arguments.

class SchechterFunction(object):

    def __init__(self, alpha, Lstar):

        #Save the input values.
        self.alpha = alpha
        self.Lstar = Lstar

        #When -2 < alpha < -1 , as often found, the integral of the GLF formally diverges or becomes negative. Instead, we only count galaxies brighter than self.Lhost_to_Lstar_min times Lstar.
        self.Lhost_to_Lstar_min = 1e-3
        return

    def P(self, Lhost_max):

        Lhost_to_Lstar_max = Lhost_max/self.Lstar

        if Lhost_to_Lstar_max<self.Lhost_to_Lstar_min:
            return 0

        return np.float(gammainc(self.alpha+1, self.Lhost_to_Lstar_min, Lhost_to_Lstar_max) / gammainc(self.alpha+1, self.Lhost_to_Lstar_min))