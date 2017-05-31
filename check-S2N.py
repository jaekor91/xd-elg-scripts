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
grzflux2 = load_grz_flux_dereddened(set2)
grzivar2 = load_grz_invar(set2)
d2m2 = load_DEEP2matched(set2) # DEEP2_matched? 
sn2 = grz_S2N(grzflux2, grzivar2)

# Field 3
set3 = load_fits_table("DECaLS-DR3-DEEP2f3-glim24.fits")
set3 = set3[reasonable_mask(set3)] # Applying reasonable mask
grz3 = load_grz(set3)
cn3 = load_cn(set3)
w3 = load_weight(set3)
grzflux3 = load_grz_flux_dereddened(set3)
grzivar3 = load_grz_invar(set3)
d2m3 = load_DEEP2matched(set3) # DEEP2_matched? 
sn3 = grz_S2N(grzflux3, grzivar3)

# Field 4
set4 = load_fits_table("DECaLS-DR3-DEEP2f4-glim24.fits")
set4 = set4[reasonable_mask(set4)] # Applying reasonable mask
grz4 = load_grz(set4)
cn4 = load_cn(set4)
w4 = load_weight(set4)
grzflux4 = load_grz_flux_dereddened(set4)
grzivar4 = load_grz_invar(set4)
d2m4 = load_DEEP2matched(set4) # DEEP2_matched? 
sn4 = grz_S2N(grzflux4, grzivar4)

# Load the intersection area
area = np.loadtxt("intersection-area-f234")

# Combine 
cn = np.concatenate((cn2,cn3,cn4))
w = np.concatenate((w2, w3, w4))
grz = combine_grz(grz2, grz3, grz4)
grzflux = combine_grz(grzflux2, grzflux3, grzflux4)
grzivar = combine_grz(grzivar2, grzivar3, grzivar4)
d2m = np.concatenate((d2m2, d2m3, d2m4))
sn = combine_grz(sn2,sn3,sn4)

print("Completed.\n")


##############################################################################
print("2. Calculate fraction of objects that have flux SN<thres=3,4,5 for each band.")
cn_list = [cn2, cn3, cn4]
sn_list = [sn2, sn3, sn4]

for thres in [3,4,5]:
    print("Fraction of objects with S/N<%.2f"%(thres))
    print("Class & band & Field 2 & Field 3 & Field 4 \\\\ \\hline")
    # For each class
    for i in range(7):
        # For each field
        g_str = [cnames[i],"g"]
        r_str = ["", "r"]
        z_str = ["", "z"]

        for field in range(3):
            iClass = cn_list[field]==i
            sn_g, sn_r, sn_z = sn_list[field]
            # For each color
            num = iClass.sum()
            g_str.append("%.2f\\%%"%(((sn_g[iClass]<thres).sum()/num) *100))
            r_str.append("%.2f\\%%"%(((sn_r[iClass]<thres).sum()/num)*100))
            z_str.append("%.2f\\%%"%(((sn_z[iClass]<thres).sum()/num) *100))

        g_str=" & ".join(g_str)+"\\\\ \\hline"
        r_str=" & ".join(r_str)+"\\\\ \\hline"
        z_str=" & ".join(z_str)+"\\\\ \\hline \\hline"
        print(g_str)
        print(r_str)
        print(z_str)
    print("\n")