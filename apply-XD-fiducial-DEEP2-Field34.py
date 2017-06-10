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
class_eff = [1., 1., 0., 0.6, 0., 0.25, 0., 0.]

##############################################################################
print("1. Load DR3-DEEP2 data. Field 3 and 4 Only.")
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
area = np.loadtxt("intersection-area-f234")[1:].sum()

# Combine 
cn = np.concatenate((cn3,cn4))
w = np.concatenate((w3, w4))
grz = combine_grz(grz3, grz4)
grzflux = combine_grz(grzflux3, grzflux4)
grzivar = combine_grz(grzivar3, grzivar4)
d2m = np.concatenate((d2m4, d2m3))
num_unmatched = (d2m==0).sum()
redz = np.concatenate((redz3, redz4))
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# Giving unmatched objects its proper number.
cn[cn<0] = 7
print("Completed.\n")

##############################################################################
param_tag = "-Field34"



##############################################################################
print("2. Compute XD projection based on fiducial set of parameters.")
param_directory = "./"
w_mag = 0.05/2.
w_cc = 0.025/2.
f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]

slices = None # np.arange(21.5, 24.0, wmag)
start = time.time()
grid, last_FoM = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
                          reg_r=5e-4,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1], param_tag2=param_tag)
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
                       rz_ref=0.5, reg_r=5e-4/(w_cc**2 * w_mag), f_i=f_i,\
                       gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1], param_tag2=param_tag)
print("Completed.\n")


##############################################################################
print("4. Compute FDR cut.")
iFDR = FDR_cut(grz)
print("Completed.\n")


##############################################################################
print("5. Print FDR cut, XD cut, XD proj. results.")
print(" & ".join(["Cut", "Type", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "FoM"]) + "\\\\ \hline")

# FDR
return_format = ["FDR", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn[iFDR], w[iFDR], area,rwd="D", num_classes=8, \
     return_format = return_format))

# XD cut
return_format = ["XD", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD], w[iXD], area,rwd="D", num_classes=8, \
     return_format = return_format))

# XD projection
return_format = ["XD", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut_grid(grid, return_format))

print("Completed.\n")


##############################################################################
print("6. Plot n(z) for the selection.")
dz = 0.05
fname = "dNdz-XD-fiducial-FDR-DEEP2-Field34.png"
plot_dNdz_selection(cn, w, iXD, redz, area, dz=0.05, \
	iselect2=iFDR, plot_total=False, fname=fname, color1="black", color2="red", color_total="green",\
	label1="XD fid.", label2="FDR", label_total="DEEP2 Total")

print("Completed.\n")


# ##############################################################################
print("7. Create many slices for a movie/stills.")
# bnd_fig_directory = "./bnd_fig_directory/XD-fiducial-Field34/"
# fname = "XD-fiducial-f34"

# print("7a. Creating stills")
# for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
# 	print("Slice %.3f"%m)
# 	XD.plot_slice(grid, m, bnd_fig_directory, fname)
# print("Completed.\n")

# print("7b. Creating a movie")
# dm = w_mag
# for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
# 	print("Index %d, Slice %.3f" % (i,m))
# 	XD.plot_slice(grid, m, bnd_fig_directory, fname, movie_tag=i)	

# print("Completed.\n")

# print("Command for creating a movie.:\n \
# 	ffmpeg -r 6 -start_number 0 -i XD-fiducial-f34-mag0-%d.png -vcodec mpeg4 -y XD-fiducial-f34-movie.mp4")



##############################################################################
print("8. To compare, compute XD projection based on fiducial set of parameters.")
param_directory = "./"
f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]

start = time.time()
grid_fiducial, last_FoM_fiducial = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
                          reg_r=5e-4,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
print("Time taken: %.2f seconds" % (time.time()-start))
print("Computed last FoM based on the grid: %.3f"%last_FoM_fiducial)

iXD_fiducial, FoM_fiducial = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM_fiducial, param_directory, \
            glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5, reg_r=5e-4/(w_cc**2 * w_mag), f_i=f_i,\
                       gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
print("Completed.\n")


##############################################################################
print("9. Create many slices for a movie/stills.")
bnd_fig_directory = "./bnd_fig_directory/XD-fiducial-Field34-Field234-comparison/"
fname = "XD-fiducial-Field34-Field234-comparison"

print("9a. Creating stills")
for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
    print("Slice %.3f"%m)
    XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname)
print("Completed.\n")

print("9b. Creating a movie")
dm = w_mag
for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
    print("Index %d, Slice %.3f" % (i,m))
    XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname, movie_tag=i)   

print("Completed.\n")

print("Command for creating a movie.:\n \
    ffmpeg -r 6 -start_number 0 -i XD-fiducial-Field34-Field234-comparison-mag0-%d.png -vcodec mpeg4 -y XD-fiducial-Field34-Field234-comparison-movie.mp4")


##############################################################################
print("10. Make dNdm plots for both grids.")
# Total 
fname = "dNdm-XD-fiducial-Field34-Field234-dNdm-Total"
XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, plot_type="Total", label1 ="Field34", \
  class_eff =  [1., 1., 0.0, 0.6, 0., 0.25, 0.])

# DESI
fname = "dNdm-XD-fiducial-Field34-Field234-dNdm-DESI"
XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, plot_type="DESI", label1 ="Field34", \
  class_eff =  [1., 1., 0.0, 0.6, 0., 0.25, 0.])
print("Completed.\n")

