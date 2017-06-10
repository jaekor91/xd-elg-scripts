# Load modules
import numpy as np
import time
from astropy.io import ascii, fits
from xd_elg_utils import *


# Constants
large_random_constant = -999119283571
deg2arcsec=3600
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "","D2unobserved"]

##############################################################################
print("1. Load DR3-DEEP2 data.")
# Field 2
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
set2 = set2[reasonable_mask(set2)] # Applying reasonable mask
grz2 = load_grz(set2)
cn2 = load_cn(set2)
w2 = load_weight(set2)
grzflux2 = load_grz_flux(set2)
grzivar2 = load_grz_invar(set2)
d2m2 = load_DEEP2matched(set2) # DEEP2_matched? 

# Field 3
set3 = load_fits_table("DECaLS-DR3-DEEP2f3-glim24.fits")
set3 = set3[reasonable_mask(set3)] # Applying reasonable mask
grz3 = load_grz(set3)
cn3 = load_cn(set3)
w3 = load_weight(set3)
grzflux3 = load_grz_flux(set3)
grzivar3 = load_grz_invar(set3)
d2m3 = load_DEEP2matched(set3) # DEEP2_matched? 

# Field 4
set4 = load_fits_table("DECaLS-DR3-DEEP2f4-glim24.fits")
set4 = set4[reasonable_mask(set4)] # Applying reasonable mask
grz4 = load_grz(set4)
cn4 = load_cn(set4)
w4 = load_weight(set4)
grzflux4 = load_grz_flux(set4)
grzivar4 = load_grz_invar(set4)
d2m4 = load_DEEP2matched(set4) # DEEP2_matched? 


# Load the intersection area
area = np.loadtxt("intersection-area-f234")

# Combine 
cn = np.concatenate((cn2,cn3,cn4))
w = np.concatenate((w2, w3, w4))
grz = combine_grz(grz2, grz3, grz4)
grzflux = combine_grz(grzflux2, grzflux3, grzflux4)
grzivar = combine_grz(grzivar2, grzivar3, grzivar4)
d2m = np.concatenate((d2m2, d2m3, d2m4))

print("Completed.\n")


##############################################################################
print("2. Put the data in appropriate form for processing.")
rz_gr=grz2rz_gr(grz)
rz_gr_covar = rz_gr_covariance(grzflux, grzivar)

print("Completed.\n")


##############################################################################
print("3. Fitting GMM with K=1,2,3 for Gold through LowZ classes.")
niter = 10
w_reg = 0.05**2
maxsnm= True # Try maximum number of splitand merge?
init_var = 0.5**2
pt_size = 3.

for i in range(6):
    print(cnames[i])

    # Sub-select the data
    ibool = cn==i
    ydata = rz_gr[ibool]; ycovar = rz_gr_covar[ibool]; weight = w[ibool]
    print("Num data points: %d" % ydata.shape[0])

    for K in range(1,4):
        if K == 3:
            niter=10
        else: 
            niter=50
 
        print("Component number K = %d" % K)
        # Make the fit
        # Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar = XD_gr_rz_fit(ydata, ycovar, weight, niter, K, init_var=init_var, w_reg = 0.05**2)

        # Import trained parameters
        Sxamp_init = np.load("%d-params-init-amps-glim24-K%d.npy"%(i,K))
        Sxcovar_init = np.load("%d-params-init-covars-glim24-K%d.npy"%(i,K))
        Sxmean_init = np.load("%d-params-init-means-glim24-K%d.npy"%(i,K))
        Sxamp = np.load("%d-params-fit-amps-glim24-K%d.npy"%(i,K))
        Sxcovar = np.load("%d-params-fit-covars-glim24-K%d.npy"%(i,K))
        Sxmean = np.load("%d-params-fit-means-glim24-K%d.npy"%(i,K))

        # Plot initial and final fits and save.
        fname = "%d-fit-cc-glim24-K%d" %(i, K)
        plot_XD_fit(ydata, weight, Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, fname=fname, pt_size=pt_size)

        # Save the best fit parameters.
        # save_params(Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, i, K, tag="")
print("Completed.\n")
        


##############################################################################
print("4. Fitting GMM with K=1,2,...7 for DEEP2reject class. Sub-sampling points.")
i = 6
pt_size = 1.5
niter = 10
maxsnm = False; snm = 3
subsample = True
init_var = 0.5**2

print(cnames[i])

# Sub-select the data
ibool = cn==i
ydata = rz_gr[ibool]; ycovar = rz_gr_covar[ibool]; weight = np.ones(ydata.shape[0])

print("Num data points: %d" % ydata.shape[0])

for K in range(1,8):
    print("Component number K = %d" % K)
    # Make the fit
    # Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar = XD_gr_rz_fit(ydata, ycovar, weight, niter, K, init_var=init_var, w_reg = 0.05**2)

    # Import trained parameters
    Sxamp_init = np.load("%d-params-init-amps-glim24-K%d.npy"%(i,K))
    Sxcovar_init = np.load("%d-params-init-covars-glim24-K%d.npy"%(i,K))
    Sxmean_init = np.load("%d-params-init-means-glim24-K%d.npy"%(i,K))
    Sxamp = np.load("%d-params-fit-amps-glim24-K%d.npy"%(i,K))
    Sxcovar = np.load("%d-params-fit-covars-glim24-K%d.npy"%(i,K))
    Sxmean = np.load("%d-params-fit-means-glim24-K%d.npy"%(i,K))
    
    # Plot initial and final fits and save.
    fname = "%d-fit-cc-glim24-K%d" %(i, K)
    plot_XD_fit(ydata, weight, Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, fname=fname, pt_size=pt_size)

    # Save the best fit parameters.
    # save_params(Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, i, K)
print("Completed.\n")









