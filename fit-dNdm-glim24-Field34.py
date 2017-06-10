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
print("1. Load DR3-DEEP2 data. Only Field 3 and 4.")
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
cn = np.concatenate((cn3,cn4))
w = np.concatenate((w3, w4))
grz = combine_grz(grz3, grz4)
grzflux = combine_grz(grzflux3, grzflux4)
grzivar = combine_grz(grzivar3, grzivar4)
d2m = np.concatenate((d2m3, d2m4))

print("Completed.\n")



##############################################################################
print("2. Specify basic fit parameters.")
# bin width
bw = 0.05
print("Bin width %.3f" % bw)

# min/max magnitude
magmin = 21.
magmax = 24.
print("min/max range [%.1f, %1f]" % (magmin, magmax))

# Number of times to try fit
niter = 10
print("# of iteration for fitting: %d " % niter)

# magnitude to fit
g = grz[0]
print("Magnitude to fit: g")

# area
area_total = area[1]+area[2] # Field 3 and 4
print("Total area used for fitting %.3f" % area_total)


##############################################################################
print("3. Fitting and saving the results in numpy format.\n \
	Figures of plots are saved as **-fit-dNdm-glim24-Field34.png")

for i in range(7):
	print(cnames[i])
	# fit
	ibool = cn == i; weight = w[ibool]; mag = g[ibool]
	fname = "%d-fit-dNdm-glim24-Field34" % i    

    # It was empirically determined that the fitting works better for class >=4 with broken_tol=1 rather than the default 1e-2.
	if i<4:
		pow_params, broken_params = dNdm_fit(mag,weight,bw,magmin, magmax,area_total, niter = niter, cn2fit=i, fname=fname)
	else:
		pow_params, broken_params = dNdm_fit(mag,weight,bw,magmin, magmax,area_total, niter = niter, cn2fit=i, broken_tol=1, fname=fname)

	# save
	fname_pow = "%d-fit-pow-glim24-Field34" % i
	np.savetxt(fname_pow, pow_params)
	fname_broken = "%d-fit-broken-glim24-Field34" % i
	np.savetxt(fname_broken, broken_params)    
	print("\n\n")


