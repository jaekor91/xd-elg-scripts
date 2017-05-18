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
num_unmatched = (d2m==0).sum()
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
print("Completed.\n")


##############################################################################
print("2. Plot the objects by class. Only g>21.")
print("Output plot filenames: cc-gr-rz-g21to24-**.png")

ibool = grz[0]>21

area_total = np.sum(area)
fname = None
for i,e in enumerate(cnames[:7]):
	print(e)
	if i <6:
		pt_size=.7
		fname = "cc-gr-rz-g24-%d%s"%(i,e)        
		plot_grz_class(grz, cn, w, area_total, pick=i,mask=ibool,fname=fname, pt_size=pt_size)
	elif i == 6:
		pt_size=0.3
		fname = "cc-gr-rz-g24-%d%s"%(i,e)        
		plot_grz_class(grz, cn, w, area_total, pick=i,mask=ibool,fname=fname, pt_size=pt_size)        
print("Completed.\n")


##############################################################################
print("3. Plot objects in all classes. Only g>21.")
print("Output plot filenames: cc-gr-rz-g21to24-all.png")
fname = "cc-gr-rz-g24-all"
plot_grz_class_all(grz, cn, w, area_total, mask=ibool,fname=fname, pt_size1=2., pt_size2=1.5)        
print("Completed.\n")

