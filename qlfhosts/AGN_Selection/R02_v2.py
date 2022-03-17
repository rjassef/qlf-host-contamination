import numpy as np

from .selection_criteria_class_v2 import SelectionCriteria

#Extremely simplified version of the SDSS selection criteria for low-z quasars by Richards et al. (2002).
class R02(SelectionCriteria):

    def __init__(self):
        super(R02,self).__init__()
        return

    def is_agn(self, mag):
        #if np.isinf([mag['sdssu'],mag['sdssg'],mag['sdssr']]).any():
        #    return False
        if mag['sdssu']-mag['sdssg']<0.9 and mag['sdssg']-mag['sdssr']<0.4:
            return True
        else:
            return False