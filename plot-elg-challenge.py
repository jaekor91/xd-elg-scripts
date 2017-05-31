# Load modules
import numpy as np
from xd_elg_utils import *
# Note that table_utils must be called after xd_elg_utils to overwrite certain
# routine definitions. I understand that this is a bad practice but for this
# non-critical application, I will condone this.
from table_utils import *
import XD_selection_module as XD
import time


# Constants
large_random_constant = -999119283571
deg2arcsec=3600
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]

##############################################################################
print("1. Load data")
# Field 2
set2 = load_fits_table("DECaLS-DR3-DEEP2f2-glim24.fits")
set2 = set2[reasonable_mask(set2)] # Applying reasonable mask
grz2 = load_grz(set2)
cn2 = load_cn(set2)
w2 = load_weight(set2)
grzflux2 = load_grz_flux_dereddened(set2)
grzivar2 = load_grz_invar(set2)
d2m2 = load_DEEP2matched(set2) # DEEP2_matched?
redz2 = load_redz(set2)
r_dev2, r_exp2 = load_shape(set2)
# W1_2, W2_2 = load_W1W2(set2)
# W1_finvar_2, W2_finvar_2 = load_W1W2_fluxinvar(set2)
# W1_flux_2, W2_flux_2 = load_W1W2_flux(set2)


# Field 3
set3 = load_fits_table("DECaLS-DR3-DEEP2f3-glim24.fits")
set3 = set3[reasonable_mask(set3)] # Applying reasonable mask
grz3 = load_grz(set3)
cn3 = load_cn(set3)
w3 = load_weight(set3)
grzflux3 = load_grz_flux_dereddened(set3)
grzivar3 = load_grz_invar(set3)
d2m3 = load_DEEP2matched(set3) # DEEP2_matched? 
redz3 = load_redz(set3)
r_dev3, r_exp3 = load_shape(set3)
# W1_3, W2_3 = load_W1W2(set3)
# W1_finvar_3, W2_finvar_3 = load_W1W2_fluxinvar(set3)
# W1_flux_3, W2_flux_3 = load_W1W2_flux(set3)



# Field 4
set4 = load_fits_table("DECaLS-DR3-DEEP2f4-glim24.fits")
set4 = set4[reasonable_mask(set4)] # Applying reasonable mask
grz4 = load_grz(set4)
cn4 = load_cn(set4)
w4 = load_weight(set4)
grzflux4 = load_grz_flux_dereddened(set4)
grzivar4 = load_grz_invar(set4)
d2m4 = load_DEEP2matched(set4) # DEEP2_matched? 
redz4 = load_redz(set4)
r_dev4, r_exp4 = load_shape(set4)
# W1_4, W2_4 = load_W1W2(set4)
# W1_finvar_4, W2_finvar_4 = load_W1W2_fluxinvar(set4)
# W1_flux_4, W2_flux_4 = load_W1W2_flux(set4)


# Load the intersection area
area = np.loadtxt("intersection-area-f234").sum()

# Combine 
cn = np.concatenate((cn2,cn3,cn4))
w = np.concatenate((w2, w3, w4))
grz = combine_grz(grz2, grz3, grz4)
grzflux = combine_grz(grzflux2, grzflux3, grzflux4)
grzivar = combine_grz(grzivar2, grzivar3, grzivar4)
d2m = np.concatenate((d2m2, d2m3, d2m4))
num_unmatched = (d2m==0).sum()
redz = np.concatenate((redz2, redz3, redz4))
# W1 = np.concatenate((W1_2,W1_3,W1_4))
# W2 = np.concatenate((W2_2,W2_3,W2_4))
# W1_finvar = np.concatenate((W1_finvar_2,W1_finvar_3,W1_finvar_4))
# W2_finvar = np.concatenate((W2_finvar_2,W2_finvar_3,W2_finvar_4))
# W1_flux = np.concatenate((W1_flux_2,W1_flux_3,W1_flux_4))
# W2_flux = np.concatenate((W2_flux_2,W2_flux_3,W2_flux_4))

# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# Giving unmatched objects its proper number.
cn[cn<0] = 7

print("Completed.\n")


##############################################################################
print("2. Define classes: \n")
print("To illustrate challenges involved: We define four classes \n \
- DEEP2 color rejected: MMT study showed that they are poor targets for DESI. \n \
    - 0, Color: Black\n \
- Things we want: Gold and Silver\n \
    - 1, Color: Orange\n \
- Things we might want but are uncertain about: NoOII and NoZ\n \
    - 2, Color: Blue\n \
- Things we don't want: LowOII and LowZ objects  \n \
    - 3, Color: Red \n\
Show how these vary over magnitudes in slices.")

# Get g-r r-z
g,r,z = grz
xrz = r-z
ygr = g-r

# Colors
colors_illustrate = ["black", "orange", "blue", "red"]

# Setting conditions
ibool1 = cn==6 # DEEP2 reject 0
ibool2 = np.logical_or((cn==0), (cn==1)) # we want 1
ibool3 = np.logical_or((cn==3), (cn==5)) # We may want 2
ibool4 = np.logical_or((cn==2), (cn==4)) # We don't want 3
print("Completed.\n")



##############################################################################
print("3. Slice plots for 0, 1, 2")
# Setting sizes
pt_size1 = 10
pt_size2 = pt_size3 = pt_size4 = 10
ft_size = 20
tick_size = 15

# Putting conditions and pt sizes in a list
ibool = [ibool1, ibool2, ibool3, ibool4]
pt_size = [pt_size1,pt_size2,pt_size3,pt_size4]

slices = [(21,22), (22,22.5), (22.5, 23), (23, 23.5), (23.5, 23.75), (23.75,24), (21,24)]

for e in slices:
    gmin, gmax = e
    imag = (g>gmin) & (g<gmax)
    fig = plt.figure(figsize=(5,5))
    for i in [1, 0, 2]:
        plt.scatter(xrz[ibool[i]&imag], ygr[ibool[i]&imag], s=pt_size[i],marker="s", c=colors_illustrate[i], edgecolors="none")
    plt.axis("equal")
    plt.xlim([-0.2, 2.0])
    plt.ylim([-0.2, 2.0])
    plt.xlabel(r"$r-z$", fontsize=ft_size)
    plt.ylabel(r"$g-r$", fontsize=ft_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.title(r"$%.2f <g< %.2f $" % (gmin, gmax), fontsize=ft_size)
    fname = "cc-gr-rz-elg-challenges-g%dto%d-012.png" % (gmin*100, gmax*100)
    plt.savefig(fname, dpi=400, bbox_inches="tight")        
    # plt.show()
    plt.close()


##############################################################################
print("4. Slice plots for 0, 1, 3")
# Setting sizes
pt_size1 = 10
pt_size2 = pt_size3 = pt_size4 = 10
ft_size = 20
tick_size = 15

# Putting conditions and pt sizes in a list
ibool = [ibool1, ibool2, ibool3, ibool4]
pt_size = [pt_size1,pt_size2,pt_size3,pt_size4]

slices = [(21,22), (22,22.5), (22.5, 23), (23, 23.5), (23.5, 23.75), (23.75,24), (21,24)]

for e in slices:
    gmin, gmax = e
    imag = (g>gmin) & (g<gmax)
    fig = plt.figure(figsize=(5,5))
    for i in [1, 0, 3]:
        plt.scatter(xrz[ibool[i]&imag], ygr[ibool[i]&imag], s=pt_size[i],marker="s", c=colors_illustrate[i], edgecolors="none")
    plt.axis("equal")
    plt.xlim([-0.2, 2.0])
    plt.ylim([-0.2, 2.0])
    plt.xlabel(r"$r-z$", fontsize=ft_size)
    plt.ylabel(r"$g-r$", fontsize=ft_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.title(r"$%.2f <g< %.2f $" % (gmin, gmax), fontsize=ft_size)
    fname = "cc-gr-rz-elg-challenges-g%dto%d-013.png" % (gmin*100, gmax*100)
    plt.savefig(fname, dpi=400, bbox_inches="tight")    
    # plt.show()
    plt.close()

print("Completed.\n")


##############################################################################
print("5. Slice plots for 0, 1, 3")
# Setting sizes
pt_size1 = 10
pt_size2 = pt_size3 = pt_size4 = 10
ft_size = 20
tick_size = 15

# Putting conditions and pt sizes in a list
ibool = [ibool1, ibool2, ibool3, ibool4]
pt_size = [pt_size1,pt_size2,pt_size3,pt_size4]

slices = [(21,22), (22,22.5), (22.5, 23), (23, 23.5), (23.5, 23.75), (23.75,24), (21,24)]

for e in slices:
    gmin, gmax = e
    imag = (g>gmin) & (g<gmax)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,10))
    for i in [1, 0, 2]:
        ax1.scatter(xrz[ibool[i]&imag], ygr[ibool[i]&imag], s=pt_size[i],marker="s", c=colors_illustrate[i], edgecolors="none")
    ax1.set_aspect("equal")
    ax1.set_xlim([-0.2, 2.0])
    ax1.set_ylim([-0.2, 2.0])
    ax1.set_xlabel(r"$r-z$", fontsize=ft_size)
    ax1.set_ylabel(r"$g-r$", fontsize=ft_size)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax1.set_title(r"$%.2f <g< %.2f $" % (gmin, gmax), fontsize=ft_size)
    
    for i in [1, 0, 3]:
        ax2.scatter(xrz[ibool[i]&imag], ygr[ibool[i]&imag], s=pt_size[i],marker="s", c=colors_illustrate[i], edgecolors="none")
    ax2.set_aspect("equal")
    ax2.set_xlim([-0.2, 2.0])
    ax2.set_ylim([-0.2, 2.0])
    ax2.set_xlabel(r"$r-z$", fontsize=ft_size)
    ax2.set_ylabel(r"$g-r$", fontsize=ft_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.set_title(r"$%.2f <g< %.2f $" % (gmin, gmax), fontsize=ft_size)

    fname = "cc-gr-rz-elg-challenges-g%dto%d.png" % (gmin*100, gmax*100)
    plt.savefig(fname, dpi=400, bbox_inches="tight")
    
    # plt.show()
    plt.close()
print("Completed.\n")



