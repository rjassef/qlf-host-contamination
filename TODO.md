- [ ] Make reddening a loadable module. For now, it is wrapped up within the Shen et al. QLF but should be independent. 

- [ ] The main iteration module can only deal with the Shen et al. and Hopkins et al. models, since the fNH module relies on the X-ray luminosity, converted from the Richards et al. (2006) SED template. 

- [ ] Create a module for the Eddington ratio version of the host contamination that uses the parametrization of [Shen et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...680..169S/abstract). May not be necessary, as they find results that agree with those of Kollmeier et al. (2006) and that the Eddington distribution depends only weakly on mass and luminosity for quasars.

- test