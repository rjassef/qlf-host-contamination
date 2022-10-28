import numpy as np

from .selection_criteria_class import SelectionCriteria

#Extremely simplified version of the SDSS selection criteria for low-z quasars by Richards et al. (2002).
class R02(SelectionCriteria):

    def __init__(self):
        #These are the bands we will need. The names need to match the filter curves in the bandpasses folder.
        bp_names = ['sdssu', 'sdssg', 'sdssr']
        super(R02,self).__init__(bp_names)
        return

    def is_agn(self, mag):
        #if np.isinf([mag['sdssu'],mag['sdssg'],mag['sdssr']]).any():
        #    return False
        if mag['sdssu']-mag['sdssg']<0.9 and mag['sdssg']-mag['sdssr']<0.4:
            return True
        else:
            return False