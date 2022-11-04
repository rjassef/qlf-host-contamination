import numpy as np
import astropy.units as u 
from astropy.constants import c, L_sun
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from regex import P
from scipy.interpolate import interp1d

from qlfhosts.GLFs import Uniform, Willmer06, Kollmeier06, Ananna22

from qlfhosts.util.magCount import MagCount

#Set the GLFs to use.
glfs = [
    Uniform,
    Willmer06,
    Kollmeier06,
    Ananna22,
]

#Start the plot.
plt.xlim([15, 28])
plt.tick_params(top=True, right=True, grid_alpha=0.5)
plt.grid()

for glf in glfs:
    Nagn_obj = MagCount(Galaxy_Luminosity_Distribution=glf)
    Ntot, m_grid = Nagn_obj.calc()

    plt.plot(m_grid, Ntot, label=str(glf.__name__))

#Finalize the plot.
plt.yscale('log')
plt.legend()
plt.xlabel("g-band magnitude")
plt.ylabel("Number counts")

#Draw the nominal 5 sigma depth of the WFD. 
plt.axvline(27.4, color='black', linestyle='dashed')
yloc = 10**(np.mean(np.log10(plt.ylim()))-1.0)
plt.text(27.0, yloc, r'WFD $5\sigma$ depth', rotation='vertical')

plt.savefig("Quasar_mag_counts_g_v3.png", dpi=200)


