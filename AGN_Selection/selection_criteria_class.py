import numpy as np

class SelectionCriteria(object):

    def __init__(self):
        return

    def is_selected(self,Lhost_Lagn, Alam, sed):
        
        #Get the magnitudes. 
        sed.get_mags(Alam, Lhost_Lagn)
   
        #Apply the selection criterion. 
        return self.is_agn(sed.mag)
