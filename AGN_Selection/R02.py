import numpy as np

from .selection_criteria_class import SelectionCriteria

#Extremely simplified version of the SDSS selection criteria for low-z quasars by Richards et al. (2002).
class R02(SelectionCriteria):

    def __init__(self):
        super(R02,self).__init__()
        return

    def is_agn(self, mag):
        """
        Assumes that mag is an array of SDSS AB magnitudes with u in the first position, g in the second and r in the third.
        """
        if np.isinf(mag).any():
            return False
        if mag[0]-mag[1]<0.9 and mag[1]-mag[2]<0.4:
            return True
        else:
            return False