import numpy as np

#Function to find the maximum tolerated ratio between the host and AGN at wavelength lambda for which the object would be selected as an AGN according to criterion sel_crit.
def Lhost_Lagn_max(agn_sed, hosts_sed, Alam, lam_rest, sel_crit, xlow=0, xhig=100, niter_max=100, debug=False):

    #Find the unscaled luminosity ratio. 
    Lnu_agn_unscaled = agn_sed.Lnu(lam_rest)
    Lnu_hosts_unscaled = hosts_sed.Lnu(lam_rest)

    #There should only be one AGN SED, but there can be multiple hosts. 
    x_unscaled = np.zeros(Lnu_hosts_unscaled.shape)
    x = np.zeros(len(x_unscaled))
    for k, Lnu_host_unscaled in enumerate(Lnu_hosts_unscaled):
        x_unscaled[k] = Lnu_agn_unscaled[0]/Lnu_host_unscaled

        #Don't go through this if the host sed template has 0 likelihood (i.e., is not being used)
        if hosts_sed.likelihood[k]==0.:
            x[k] = 0.
            continue

        #Check that xmin selects the source and that xmax does not. If this is not met, then return the bound.
        if not sel_crit.is_selected(xlow, Alam, lam_rest, agn_sed, hosts_sed, k, x_unscaled):
            if debug:
                print("Object not selected as AGN at xlow. Returning xlow.")
            x[k] = xlow
            continue

        if sel_crit.is_selected(xhig, Alam, lam_rest, agn_sed, hosts_sed, k, x_unscaled):
            if debug:
                print("Object selected as AGN at xhig. Returning xhig.")
            x[k] = xhig
            continue

        #Otherwise, bisect the way to the right solution.
        x1 = xlow
        x2 = xhig
        for i in range(niter_max):
            x[k] = 0.5*(x1+x2)
            if sel_crit.is_selected(x[k], Alam, lam_rest, agn_sed, hosts_sed, k, x_unscaled):
                x1 = x[k]
            else:
                x2 = x[k]
            if (x1-x2)/x[k]<1e-4:
                break

        #Only gets here if solution not found in niter_max iterations.
        if debug:
            print("Exceed niter_max={} iterations. Returning current solution.".format(niter_max))
        continue

    #print(x)
    return x

        

