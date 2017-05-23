##############################################################################
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
# Introduction: Once having downloaded the git repository from
#   https://github.com/jaekor91/xd-elg-scripts.git
# the user can run this script to make projections for how XD algorithm will 
# peform when used to select DESI ELG sample. The user must XD parameters
# and which outputs are wanted.
#
# Required data: README contains instruction for downloading required data
# for this section.
# 
# To run: In xd-elg-scripts directory, type "python XD-user.py".
# 
# Note: This script can be used to make upto two projections based on two different
# sets of parameters. When making comparisons, the projection based on the second
# set is considered as a reference.
# 
##############################################################################
# --- 1 or 2 projections? --- #
# If True, two projections based on two sets of parameters below are computed
# and compared.
two_projections = True
# 
##############################################################################
# --- Outputs --- #
# dNdm plots

# dNdm movie

# 

# dNdz plots






##############################################################################
# --- XD projection 1 Parameters --- #
# This is a directory where all XD parameters required for projections are
# stored. See README for download instruction.
param_directory = "./XD-parameters/"

# For each class, choose whether to use power law (0) or broken power law (1).
# Recall: 0-Gold, 1-Silver, 2-LowOII, 3-NoOII, 4-LowZ, 5-NoZ, 6-D2reject
dNdm_type = [1, 1, 0, 1, 0, 0, 1]

# For each class, choose how many component gaussians to use for GMM
# Recall: 0-Gold, 1-Silver, 2-LowOII, 3-NoOII, 4-LowZ, 5-NoZ, 6-D2reject
K_i = [2,2,2,3,2,2,7]

# The limiting magnitude depths.
glim=23.8
rlim=23.4
zlim=22.4

# Total number of fibers to be used.
N_tot=2400

# Figure of Merit (FoM) weights. Recall FoM = (sum_j f_j * n_j ) / (sum_i n_i)
# Note that this is different "class efficiency" which we define as
# the fraction of objects in each class we expect to be good objects
# for DESI.
f_i=[1., 1., 0., 0.25, 0., 0.25, 0.]

# Reference points based on which the number density conserving noise are
# calculated.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
gr_ref=0.5
rz_ref=0.5

# Regularizing parameter to be added to the denomitor when calculating FoM.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
reg_r=5e-4

# Grid parameters. The finer grid will slowdown the calculation but may 
# give marginal-to-somewhat more accurate result.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
w_mag = 0.05/2.
w_cc = 0.025/2.
minmag = 21.5+w_mag/2.
maxmag = 24.

# XD selection min/max g-magnitude when applied to DEEP2 data
gmin = 21.
gmax = 24.

# Class efficiency: Expected fraction of good objects in the class.
gold_eff = 1.
silver_eff = 1.
NoOII_eff = 0.6
NoZ_eff = 0.25


##############################################################################
# --- XD projection 2 Parameters --- #
# Second set parameters.
# This is a directory where all XD parameters required for projections are
# stored. See README for download instruction.
param_directory2 = "./XD-parameters/"

# For each class, choose whether to use power law (0) or broken power law (1).
# Recall: 0-Gold, 1-Silver, 2-LowOII, 3-NoOII, 4-LowZ, 5-NoZ, 6-D2reject
dNdm_type2 = [1, 1, 0, 1, 0, 0, 1]

# For each class, choose how many component gaussians to use for GMM
# Recall: 0-Gold, 1-Silver, 2-LowOII, 3-NoOII, 4-LowZ, 5-NoZ, 6-D2reject
K_i2 = [2,2,2,3,2,2,7]

# The limiting magnitude depths.
glim2=23.8
rlim2=23.4
zlim2=22.4

# Total number of fibers to be used.
N_tot2=2400

# Figure of Merit (FoM) weights. Recall FoM = (sum_j f_j * n_j ) / (sum_i n_i)
# Note that this is different "class efficiency" which we define as
# the fraction of objects in each class we expect to be good objects
# for DESI.
f_i2=[1., 1., 0., 0.25, 0., 0.25, 0.]

# Reference points based on which the number density conserving noise are
# calculated.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
gr_ref2=0.5
rz_ref2=0.5

# Regularizing parameter to be added to the denomitor when calculating FoM.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
reg_r2=5e-4

# Grid parameters. The finer grid will slowdown the calculation but may 
# give marginal-to-somewhat more accurate result.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
w_mag2 = 0.05/2.
w_cc2 = 0.025/2.
minmag2 = 21.5+w_mag/2.
maxmag2 = 24.

# XD selection min/max g-magnitude when applied to DEEP2 data
gmin2 = 21.
gmax2 = 24.

# Class efficiency: Expected fraction of good objects in the class.
gold_eff2 = 1.
silver_eff2 = 1.
NoOII_eff2 = 0.6
NoZ_eff2 = 0.25



##############################################################################
# Below is the part of the program that produces results. 
##############################################################################
print("Load DR3-DEEP2 data.")
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
oii2 = load_oii(set2)


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
oii3 = load_oii(set3)

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
oii4 = load_oii(set4)


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
oii = np.concatenate((oii2, oii3, oii4))
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# print("Fraction of expected good objects with the rising OII threshold due \n \
# to the increased fiber number and shorter exposure time.")
iGoldSilver = np.logical_or((cn==0), (cn==1))
oii_goldsilver = oii[iGoldSilver]*1e17
w_goldsilver = w[iGoldSilver]


DESI_frac = frac_above_new_oii(oii_goldsilver, w_goldsilver, new_oii_lim(N_tot, 2400))
if two_projections:	
	DESI_frac2 = frac_above_new_oii(oii_goldsilver, w_goldsilver, new_oii_lim(N_tot2, 2400))


# Giving unmatched objects its proper number.
cn[cn<0] = 7
print("Completed.\n")




##############################################################################
print("Compute XD projection 1.")
start = time.time()
grid, last_FoM = XD.generate_XD_selection(param_directory, glim=glim, rlim=rlim, zlim=zlim, \
                          gr_ref=gr_ref, rz_ref=rz_ref, N_tot=N_tot, f_i=f_i, \
                          reg_r=reg_r,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = minmag, \
                          maxmag = maxmag, K_i = K_i, dNdm_type = dNdm_type)
print("Time taken: %.2f seconds" % (time.time()-start))
print("Computed last FoM based on the grid: %.3f"%last_FoM)
print("Completed.\n")

if two_projections:
	print("Compute XD projection 2.")
	start = time.time()
	grid2, last_FoM2 = XD.generate_XD_selection(param_directory2, glim=glim2, rlim=rlim2, zlim=zlim2, \
	                          gr_ref=gr_ref2, rz_ref=rz_ref2, N_tot=N_tot2, f_i=f_i2, \
	                          reg_r=reg_r2,zaxis="g", w_cc = w_cc2, w_mag = w_mag2, minmag = minmag2, \
	                          maxmag = maxmag2, K_i = K_i2, dNdm_type = dNdm_type2)
	print("Time taken: %.2f seconds" % (time.time()-start))
	print("Computed last FoM based on the grid: %.3f"%last_FoM2)
	print("Completed.\n")	




##############################################################################
# Unpack variables
g,r,z = grz
givar, rivar, zivar = grzivar
gflux, rflux, zflux = grzflux
print("Apply XD selection 1 to DEEP2-DECaLS data.")
iXD, FoM = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM, param_directory, \
                        glim=glim, rlim=rlim, zlim=zlim, gr_ref=gr_ref,\
                       rz_ref=rz_ref, reg_r=reg_r/(w_cc**2 * w_mag), f_i=f_i,\
                       gmin = gmin, gmax = gmax, K_i = K_i, dNdm_type = dNdm_type)
print("Completed.\n")

if two_projections:
	print("Apply XD selection 2 to DEEP2-DECaLS data.")
	iXD2, FoM2 = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM, param_directory2, \
	                        glim=glim2, rlim=rlim2, zlim=zlim2, gr_ref=gr_ref2,\
	                       rz_ref=rz_ref2, reg_r=reg_r2/(w_cc2**2 * w_mag2), f_i=f_i2,\
	                       gmin = gmin2, gmax = gmax2, K_i = K_i2, dNdm_type = dNdm_type2)
	print("Completed.\n")


##############################################################################
print("Compute FDR cut.")
iFDR = FDR_cut(grz)
print("Completed.\n")


##############################################################################
print("Print projection results in tabular form.")
print(" & ".join(["Cut", "Type", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "FoM"]) + "\\\\ \hline")

# FDR
return_format = ["FDR", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn[iFDR], w[iFDR], area,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD cut
return_format = ["XD1", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD], w[iXD], area,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD projection
return_format = ["XD1", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut_grid(grid, return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0.]))

if two_projections:
	# XD cut
	return_format = ["XD2", "Avg.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM2),  "\\\\ \hline"]
	print(class_breakdown_cut(cn[iXD2], w[iXD2], area,rwd="D", num_classes=8, \
	     return_format = return_format, class_eff = [gold_eff2*DESI_frac, gold_eff2*DESI_frac, 0.0, NoOII_eff2*DESI_frac, 0., NoZ_eff2*DESI_frac, 0., 0.]))

	# XD projection
	return_format = ["XD2", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM2),  "\\\\ \hline"]
	print(class_breakdown_cut_grid(grid2, return_format, class_eff = [gold_eff2*DESI_frac, gold_eff2*DESI_frac, 0.0, NoOII_eff2*DESI_frac, 0., NoZ_eff2*DESI_frac, 0.]))

print("Completed.\n")


# ##############################################################################
# print("6. Create many slices for a movie/stills.")
# # bnd_fig_directory = "./bnd_fig_directory/XD-Ntot3000/"
# # fname = "XD-Ntot3000"

# # print("6a. Creating stills")
# # for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
# #     print("Slice %.3f"%m)
# #     XD.plot_slice(grid, m, bnd_fig_directory, fname)
# # print("Completed.\n")

# # print("6b. Creating a movie")
# # dm = w_mag
# # for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
# #     print("Index %d, Slice %.3f" % (i,m))
# #     XD.plot_slice(grid, m, bnd_fig_directory, fname, movie_tag=i)   

# # print("Completed.\n")

# # print("Command for creating a movie.:\n \
# #     ffmpeg -r 6 -start_number 0 -i XD-Ntot3000-mag0-%d.png -vcodec mpeg4 -y XD-Ntot3000-movie.mp4")


# ##############################################################################
# print("7. To compare, compute XD projection based on fiducial set of parameters.")
# param_directory = "./"
# f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]

# start = time.time()
# grid_fiducial, last_FoM_fiducial = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
#                           gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
#                           reg_r=5e-4,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
#                           maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
# print("Time taken: %.2f seconds" % (time.time()-start))
# print("Computed last FoM based on the grid: %.3f"%last_FoM)

# iXD_fiducial, FoM_fiducial = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM_fiducial, param_directory, \
#             glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
#                        rz_ref=0.5, reg_r=5e-4/(w_cc**2 * w_mag), f_i=f_i,\
#                        gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])
# print("Completed.\n")

# ##############################################################################
# print("8. Plot n(z) for the selection.")
# dz = 0.05
# fname = "dNdz-XD-Ntot3000-fiducial-DEEP2.png"
# plot_dNdz_selection(cn, w, iXD, redz, area, dz=0.05,\
#     iselect2=iXD_fiducial, plot_total=False, fname=fname, color1="blue", color2="black", color_total="green",\
#     label1="XD Ntot3000", label2="XD fid.", gold_eff = DESI_frac, silver_eff = DESI_frac, NoOII_eff = DESI_frac*0.6, \
#     NoZ_eff = DESI_frac*0.25)
# print("Completed.\n")


# ##############################################################################
# print("9. Create many slices for a movie/stills.")
# # bnd_fig_directory = "./bnd_fig_directory/XD-Ntot3000-fiducial-comparison/"
# # fname = "XD-Ntot3000-fiducial-comparison"

# # print("9a. Creating stills")
# # for m in [22., 22.5, 23.0, 23.5, 23.75, 23.825]:
# #     print("Slice %.3f"%m)
# #     XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname)
# # print("Completed.\n")

# # print("9b. Creating a movie")
# # dm = w_mag
# # for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
# #     print("Index %d, Slice %.3f" % (i,m))
# #     XD.plot_slice_compare(grid, grid_fiducial, m, bnd_fig_directory, fname, movie_tag=i)   

# # print("Completed.\n")

# # print("Command for creating a movie.:\n \
# #     ffmpeg -r 6 -start_number 0 -i XD-Ntot3000-fiducial-comparison-mag0-%d.png -vcodec mpeg4 -y XD-Ntot3000-fiducial-comparison-movie.mp4")


# ##############################################################################
# print("10. Make dNdm plots for both grids.")
# # Total 
# fname = "dNdm-XD-Ntot3000-fiducial-dNdm-Total"
# XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, type="Total", label1 ="Ntot3000", \
# 	class_eff =  [1.*DESI_frac, 1.*DESI_frac, 0.0, 0.6*DESI_frac, 0., 0.25*DESI_frac, 0.])

# # DESI
# fname = "dNdm-XD-Ntot3000-fiducial-dNdm-DESI"
# XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, type="DESI", label1 ="Ntot3000", \
# 	class_eff =  [1.*DESI_frac, 1.*DESI_frac, 0.0, 0.6*DESI_frac, 0., 0.25*DESI_frac, 0.])
# print("Completed.\n")
