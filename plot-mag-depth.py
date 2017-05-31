# Load modules
import numpy as np
from astropy.io import fits
import numpy.lib.recfunctions as rec
from xd_elg_utils import *

large_random_constant = -9991191
deg2arcsec=3600


##############################################################################  
print("1. Load combined Tractor file with the following processing:\n\
- Concatenate all relevant Tractor files in each field\n\
- Brick primary and positive inverse variance condition (This is just getting rid of bad objects.)\n\
- Append Tycho2 mask but do not enforce them. (This mask is applied this before matching with DEEP2\n\
catalogs so as not to get spurious matches.)\n\
- Objects lie within DEEP2 window function (so that we are concerend with the same spectroscopic area)\n\
\n\
Note: De-reddened flux values are used.")

# Field 2
set2 = load_fits_table('DECaLS-DR3-Tractor-DEEP2f2.fits')
grzflux2 = load_grz_flux_dereddened(set2)
set2 = set2[is_grzflux_pos(grzflux2)]
grzflux2 = load_grz_flux_dereddened(set2)
grz2 = load_grz(set2)
grzivar2 = load_grz_invar(set2)
r_dev2, r_exp2 = load_shape(set2)

# Field 3
set3 = load_fits_table('DECaLS-DR3-Tractor-DEEP2f3.fits')
grzflux3 = load_grz_flux_dereddened(set3)
set3 = set3[is_grzflux_pos(grzflux3)]
grzflux3 = load_grz_flux_dereddened(set3)
grz3 = load_grz(set3)
grzivar3 = load_grz_invar(set3)
r_dev3, r_exp3 = load_shape(set3)


# Field 4
set4 = load_fits_table('DECaLS-DR3-Tractor-DEEP2f4.fits')
grzflux4 = load_grz_flux_dereddened(set4)
set4 = set4[is_grzflux_pos(grzflux4)]
grzflux4 = load_grz_flux_dereddened(set4)
grz4 = load_grz(set4)
grzivar4 = load_grz_invar(set4)
r_dev4, r_exp4 = load_shape(set4)

r_exp_list = [r_exp2,r_exp3,r_exp4]
grzflux_list = [grzflux2,grzflux3,grzflux4]
grzinvar_list = [grzivar2,grzivar3,grzivar4]
r_exp_min = 0.35
r_exp_max = 0.55

print("Completed.\n")


##############################################################################  
print("2. Make depth plots based on r_exp [0.35, 0.55] objects. The objects \n\
must possess positive flux invar and positive flux.")
ft_size = 20
tick_size = 15
lw =2 

dm=0.05
bins = np.arange(22,25.2,dm)
for i in range(3):
    r_exp = r_exp_list[i]
    gf, rf, zf = grzflux_list[i]
    gi, ri, zi = grzinvar_list[i]
    gf_err, rf_err, zf_err = grz_flux_error(grzinvar_list[i])

    ibool = (r_exp>r_exp_min)&(r_exp<r_exp_max)
    
    fig = plt.figure(figsize=(12,7))
    
    g_depth = mag_depth_Xsigma(gf_err[ibool])
    r_depth = mag_depth_Xsigma(rf_err[ibool])
    z_depth = mag_depth_Xsigma(zf_err[ibool])
    plt.hist(g_depth,bins=bins,histtype ="stepfilled",color="green", lw=lw,alpha=0.25)
    plt.hist(r_depth,bins=bins,histtype ="stepfilled",color="red", lw=lw,alpha=0.25)
    plt.hist(z_depth,bins=bins,histtype ="stepfilled",color="purple", lw=lw,alpha=0.25)
    plt.xlabel("mag depth", fontsize=ft_size)
    plt.ylabel("dN/d(%.2fm)"%dm, fontsize=ft_size)
    # Required
    plt.axvline(x=24.0, c="green",lw=lw,ls="--")
    plt.axvline(x=23.4, c="red",lw=lw,ls="--")
    plt.axvline(x=22.5, c="purple",lw=lw,ls="--")
    # Median
    plt.axvline(x=np.median(g_depth), c="green",lw=lw)
    plt.axvline(x=np.median(r_depth), c="red",lw=lw)
    plt.axvline(x=np.median(z_depth), c="purple",lw=lw)
    
    plt.title("Field %d"%(i+2), fontsize=ft_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    
    plt.savefig("mag-depth-field%d.png"%(i+2),bbox_inches="tight",dpi=400)
    # plt.show()
    plt.close()
print("Completed.\n")
    