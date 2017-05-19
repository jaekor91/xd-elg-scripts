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
redz2 = load_redz(set2)

# Field 3
set3 = load_fits_table("DECaLS-DR3-DEEP2f3-glim24.fits")
set3 = set3[reasonable_mask(set3)] # Applying reasonable mask
grz3 = load_grz(set3)
cn3 = load_cn(set3)
w3 = load_weight(set3)
grzflux3 = load_grz_flux(set3)
grzivar3 = load_grz_invar(set3)
d2m3 = load_DEEP2matched(set3) # DEEP2_matched? 
redz3 = load_redz(set3)

# Field 4
set4 = load_fits_table("DECaLS-DR3-DEEP2f4-glim24.fits")
set4 = set4[reasonable_mask(set4)] # Applying reasonable mask
grz4 = load_grz(set4)
cn4 = load_cn(set4)
w4 = load_weight(set4)
grzflux4 = load_grz_flux(set4)
grzivar4 = load_grz_invar(set4)
d2m4 = load_DEEP2matched(set4) # DEEP2_matched? 
redz4 = load_redz(set4)

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
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# Giving unmatched objects its proper number.
cn[cn<0] = 7
print("Completed.\n")


##############################################################################
print("2. Compute XD projection based on fiducial set of parameters but with N_tot = 3000.")
param_directory = "./"
w_mag = 0.05/2.
w_cc = 0.025/2.
f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]
N_tot = 3000

slices = None # np.arange(21.5, 24.0, wmag)
start = time.time()
grid, last_FoM = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=N_tot, f_i=f_i, \
                          reg_r=1e-4,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
print("Time taken: %.2f seconds" % (time.time()-start))
print("Computed last FoM based on the grid: %.3f"%last_FoM)
print("Completed.\n")



##############################################################################
print("3. Calculate XD cut.")
param_directory = "./" # Directory where the parameters are saved.
# last_FoM = 0.502
# Unpack variables
g,r,z = grz
givar, rivar, zivar = grzivar
gflux, rflux, zflux = grzflux

iXD, FoM = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM, param_directory, \
                        glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5, reg_r=1e-4/(w_cc**2 * w_mag), f_i=f_i,\
                       gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
print("Completed.\n")


##############################################################################
print("4. Compute FDR cut.")
iFDR = FDR_cut(grz)
print("Completed.\n")


##############################################################################
print("3. Print FDR cut, XD cut, XD proj. results.")
print(" & ".join(["Cut", "Type", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "FoM"]) + "\\\\ \hline")

# FDR
return_format = ["FDR", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn[iFDR], w[iFDR], area,rwd="D", num_classes=8, \
     return_format = return_format,\
     class_eff = [1., 1., 0.25, 0.25, 0., 0., 0., 0.]
     ))

# XD cut
return_format = ["XD", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD], w[iXD], area,rwd="D", num_classes=8, \
     return_format = return_format,\
     class_eff = [1., 1., 0.25, 0.25, 0., 0., 0., 0.]
     ))

# XD projection
return_format = ["XD", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut_grid(grid, return_format, class_eff = [1., 1., 0.25, 0.25, 0., 0., 0.]))

print("Completed.\n")


##############################################################################
print("4. Plot n(z) for the selection.")
dz = 0.05
fname = "dNdz-XD-Ntot3000-DEEP2-Total.png"
plot_dNdz_selection(cn, w, iXD, redz, area, dz=0.05, gold_eff=1, silver_eff=1, NoZ_eff=0.25, NoOII_eff=0.25,\
    iselect2=None, plot_total=True, fname=fname, color1="black", color2="red", color_total="green",\
    label1="FDR", label2="", label_total="DEEP2 Total")

print("Completed.\n")


##############################################################################
print("5. Create many slices for a movie/stills.")
bnd_fig_directory = "./bnd_fig_directory/XD-Ntot3000/"
fname = "XD-Ntot3000"

print("5a. Creating stills")
for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
    print("Slice %.3f"%m)
    XD.plot_slice(grid, m, bnd_fig_directory, fname)
print("Completed.\n")

print("5b. Creating a movie")
dm = w_mag
for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
    print("Index %d, Slice %.3f" % (i,m))
    XD.plot_slice(grid, m, bnd_fig_directory, fname, movie_tag=i)   

print("Completed.\n")

print("Command for creating a movie.:\n \
    ffmpeg -r 6 -start_number 0 -i XD-Ntot3000-mag0-%d.png -vcodec mpeg4 -y XD-Ntot3000-movie.mp4")


##############################################################################
print("6. To compare, compute XD projection based on fiducial set of parameters.")
param_directory = "./"
f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]

slices = None # np.arange(21.5, 24.0, wmag)
start = time.time()
grid_fiducial, last_FoM_fiducial = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
                          reg_r=1e-4,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
print("Time taken: %.2f seconds" % (time.time()-start))
print("Computed last FoM based on the grid: %.3f"%last_FoM)
print("Completed.\n")


##############################################################################
print("7. Create many slices for a movie/stills .")
bnd_fig_directory = "./bnd_fig_directory/XD-Ntot3000-fiducial-comparison/"
fname = "XD-Ntot3000-fiducial-comparison"

print("7a. Creating stills")
for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
    print("Slice %.3f"%m)
    XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname)
print("Completed.\n")

print("7b. Creating a movie")
dm = w_mag
for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
    print("Index %d, Slice %.3f" % (i,m))
    XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname, movie_tag=i)   

print("Completed.\n")

print("Command for creating a movie.:\n \
    ffmpeg -r 6 -start_number 0 -i XD-Ntot3000-fiducial-comparison-mag0-%d.png -vcodec mpeg4 -y XD-Ntot3000-fiducial-comparison-movie.mp4")