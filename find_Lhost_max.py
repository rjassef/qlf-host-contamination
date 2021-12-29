#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from R02 import R02
from lrt_SED_model import lrt_SED_model as SED_model
from Lhost_Lagn_max import Lhost_Lagn_max

#Create the SED model. We will be evaluating the SEDs at wavelength lam.
z = 0.3
host_type = np.array([0., 1., 0.])
lam = 0.51
agn = SED_model(z, host_type, lam)

#Set the selection criterion to the very simplified criterion by Richards et al. (2002)
sel_crit = R02()

#Set the grid in ebv we will go through.
ebv_grid = np.logspace(0., 0.5, 20)-1.0

#Run through all the ebv values finding the max host to AGN ratio that allows for the object to be selected as an AGN.
Lh_La_max = np.zeros(ebv_grid.shape)
for k,ebv in enumerate(ebv_grid):
    Lh_La_max[k] = Lhost_Lagn_max(agn, ebv, sel_crit) 

#Interpolate and plot the results.
f = interp1d(ebv_grid, Lh_La_max, kind='quadratic')
ebv_interp = np.arange(0.0, np.max(ebv_grid), 0.01)
plt.plot(ebv_interp, f(ebv_interp), 'k-')
plt.plot(ebv_grid, Lh_La_max, 'bo')
plt.show()