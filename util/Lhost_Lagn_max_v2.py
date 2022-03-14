import numpy as np

#Function to find the maximum tolerated ratio between the host and AGN at wavelength lambda for which the object would be selected as an AGN according to criterion sel_crit.
def Lhost_Lagn_max(agn_sed, hosts_sed, Alam, lam_rest, sel_crit, xlow=0, xhig=100, niter_max=100, debug=False):

    #Find the unscaled luminosity ratio. 
    Lnu_agn_unscaled = agn_sed.Lnu(lam_rest)
    Lnu_hosts_unscaled = hosts_sed.Lnu(lam_rest)

    #There should only be one AGN SED, but there can be multiple hosts. 
    x_unscaled = np.zeros(Lnu_hosts_unscaled.shape)
    for k, Lnu_host_unscaled in enumerate(Lnu_hosts_unscaled):
        x_unscaled[k] = Lnu_agn_unscaled[0]/Lnu_host_unscaled

    #Check that xmin selects the source and that xmax does not. If this is not met, then return the bound.
    if not sel_crit.is_selected(xlow, Alam, agn):
        if debug:
            print("Object not selected as AGN at xlow. Returning xlow.")
        return xlow
    if sel_crit.is_selected(xhig, Alam, agn):
        if debug:
            print("Object selected as AGN at xhig. Returning xhig.")
        return xhig

    #Otherwise, bisect the way to the right solution.
    for i in range(niter_max):
        x = 0.5*(xlow+xhig)
        if sel_crit.is_selected(x, Alam, agn):
            xlow = x
        else:
            xhig = x
        if (xhig-xlow)/x<1e-4:
            return x

    #Only gets here if solution not found in niter_max iterations.
    if debug:
        print("Exceed niter_max={} iterations. Returning current solution.".format(niter_max))
    return x

    def get_mags(x, x_unscaled, agn_sed, host_sed):
        #NEed to propagate the reddening. 
        m_agn = agn_sed.mag_unscaled
        

