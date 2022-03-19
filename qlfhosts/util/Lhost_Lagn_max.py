import numpy as np
import sys 

#Function to find the maximum tolerated ratio between the host and AGN at wavelength lambda for which the object would be selected as an AGN according to criterion sel_crit.
def Lhost_Lagn_max(agn_sed, hosts_sed, Alam, Aband_to_Alam, sel_crit, x_unscaled_all, xlow=0, xhig=100, niter_max=100, debug=False):

    #Get the AGN unscaled magnitudes.
    agn_mags = dict()
    for j, bp_name in enumerate(agn_sed.bp_names):
        agn_mags[bp_name] = agn_sed.mag_unscaled[0][j] + Aband_to_Alam[bp_name]*Alam

    #Get the host unscaled magnitudes and the difference with the AGN ones. 
    all_hosts_mag_diff = list()
    try:
        for k in range(len(hosts_sed.mag_unscaled)):
            all_hosts_mag_diff.append(dict())
            for j, bp_name in enumerate(hosts_sed.bp_names):
                all_hosts_mag_diff[-1][bp_name] = hosts_sed.mag_unscaled[k][j] - agn_mags[bp_name]
    except KeyError:
        print("Need to use the same bands in AGN and hosts SEDs.")
        sys.exit()

    #There should only be one AGN SED, but there can be multiple hosts, so lets iterate through them to get x. 
    x = np.zeros(len(hosts_sed.sps))
    for k, x_unscaled in enumerate(x_unscaled_all):

        #Don't go through this if the host sed template has 0 likelihood (i.e., is not being used)
        if hosts_sed.likelihood[k]==0.:
            x[k] = 0.
            continue

        #Isolate only the mag_diff values needed.
        mag_diff = all_hosts_mag_diff[k]

        #Check that xmin selects the source and that xmax does not. If this is not met, 
        if not sel_crit.is_selected(xlow, x_unscaled, agn_mags, mag_diff):
            if debug:
                print("Object not selected as AGN at xlow. Returning xlow.")
            x[k] = xlow
            continue

        if sel_crit.is_selected(xhig, x_unscaled, agn_mags, mag_diff):
            if debug:
                print("Object selected as AGN at xhig. Returning xhig.")
            x[k] = xhig
            continue

        #Otherwise, bisect the way to the right solution.
        x1 = xlow
        x2 = xhig
        for i in range(niter_max):
            x[k] = 0.5*(x1+x2)
            if sel_crit.is_selected(x[k], x_unscaled, agn_mags, mag_diff):
                x1 = x[k]
            else:
                x2 = x[k]
            if (x2-x1)/x[k]<1e-4:
                break

        #Only gets here if solution not found in niter_max iterations.
        if debug:
            print("Exceed niter_max={} iterations. Returning current solution.".format(niter_max))
        continue

    return x

        

