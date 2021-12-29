import numpy as np
from scipy import special

class SchechterFunction(object):

    def __init__(self, alpha, Lstar):
        self.alpha = alpha
        self.Lstar = Lstar
        return

    def P(self, Lhost_max):
        return special.gammainc(self.alpha+1, Lhost_max/self.Lstar)