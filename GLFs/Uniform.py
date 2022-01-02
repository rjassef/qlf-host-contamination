import numpy as np 

class Uniform(object):

    def P(self, Lhost_max):
        return np.ones(Lhost_max.shape)