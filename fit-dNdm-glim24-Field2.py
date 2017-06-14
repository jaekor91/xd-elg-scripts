# Load modules
import numpy as np
import time
from astropy.io import ascii, fits
from xd_elg_utils import *
# Note that table_utils must be called after xd_elg_utils to overwrite certain
# routine definitions.
from table_utils import *



# Constants
large_random_constant = -999119283571
deg2arcsec=3600
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "","D2unobserved"]

##############################################################################
print("1. Load DR3-DEEP2 data. Only Field 2 data.")
area_2 = np.loadtxt("intersection-area-f234")[0]

# Field 2
# Without both
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
set2 = set2[reasonable_mask(set2, SN=False, decam_mask="")] # Do not apply signal to noise
cn2 = load_cn(set2)
w2 = load_weight(set2)
return_format = ["ALL", "No SN/ALL", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", "\\\\ \hline"]
print(class_breakdown_cut(cn2, w2, area_2, rwd="D", num_classes=8, return_format = return_format))

# Without SN
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
set2 = set2[reasonable_mask(set2, SN=False)] # Do not apply signal to noise
cn2 = load_cn(set2)
w2 = load_weight(set2)
return_format = ["ALL", "No SN    ", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", "\\\\ \hline"]
print(class_breakdown_cut(cn2, w2, area_2, rwd="D", num_classes=8, return_format = return_format))

# Without allmask
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
set2 = set2[reasonable_mask(set2, decam_mask="")] # Do not apply signal to noise
cn2 = load_cn(set2)
w2 = load_weight(set2)
return_format = ["ALL", "No ALL   ", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", "\\\\ \hline"]
print(class_breakdown_cut(cn2, w2, area_2, rwd="D", num_classes=8, return_format = return_format))

# With both
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
return_format = ["ALL", "BOTH     ", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", "\\\\ \hline"]
set2 = set2[reasonable_mask(set2)] # Do not apply signal to noise
cn2 = load_cn(set2)
w2 = load_weight(set2)
print(class_breakdown_cut(cn2, w2, area_2, rwd="D", num_classes=8, return_format = return_format))

print("Using both DECAM_ALLMASK and SN>2.")
grz2 = load_grz(set2)
grzflux2 = load_grz_flux(set2)
grzivar2 = load_grz_invar(set2)
d2m2 = load_DEEP2matched(set2) # DEEP2_matched?  

# Load the intersection area

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
g = grz2[0]
print("Magnitude to fit: g")

# area
print("Total area used for fitting %.3f" % area_2)


##############################################################################
print("3. Fitting and saving the results in numpy format.\n \
	Figures of plots are saved as **-fit-dNdm-glim24-Field2.png")

for i in range(7):
	print(cnames[i])
	# fit
	ibool = cn2 == i; weight = w2[ibool]; mag = g[ibool]
	fname = "%d-fit-dNdm-glim24-Field2" % i    

    # It was empirically determined that the fitting works better for class >=4 with broken_tol=1 rather than the default 1e-2.
	if i<4:
		pow_params, broken_params = dNdm_fit(mag,weight,bw,magmin, magmax, area_2, niter = niter, cn2fit=i, fname=fname)
	else:
		pow_params, broken_params = dNdm_fit(mag,weight,bw,magmin, magmax, area_2, niter = niter, cn2fit=i, broken_tol=1, fname=fname)

	# save
	fname_pow = "%d-fit-pow-glim24-Field2" % i
	np.savetxt(fname_pow, pow_params)
	fname_broken = "%d-fit-broken-glim24-Field2" % i
	np.savetxt(fname_broken, broken_params)    
	print("\n\n")


