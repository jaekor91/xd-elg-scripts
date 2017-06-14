##############################################################################
# Load modules
import numpy as np
from xd_elg_utils import *
# Note that table_utils must be called after xd_elg_utils to overwrite certain
# routine definitions.
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
# peform when used to select DESI ELG sample. The user must specify the directory
# for XD parameters and which outputs are wanted.
#
# Required data: README contains instruction for downloading required data
# for this section.
# 
# To run: In xd-elg-scripts directory, type "python XD-user.py".
# 
# Note: This script can be used to make upto two projections based on two different
# sets of parameters. When making comparisons, the projection based on the second
# set is considered as reference.


##############################################################################
# --- 1 or 2 projections? --- #
# If True, two projections based on two sets of parameters below are computed
# and compared.
two_projections = False


##############################################################################
# --- Outputs --- #
# XD1 for projection based on first set and XD2 corresponding to the second set

# XD1: Boundary plots 
plot_bnd = True
plot_bnd_movie = False # Generate many slices for a movie.
bnd_fig_directory = "./bnd_fig_directory/XD1-bnd/"
bnd_fname = "XD1-bnd"
mag_slices = [22., 22.5, 23.0, 23.5, 23.75, 23.825]

# XD2: Boundary plots
plot_bnd2 = True
plot_bnd_movie2 = False # Generate many slices for a movie.
bnd_fig_directory2 = "./bnd_fig_directory/XD2-bnd/"
bnd_fname2 = "XD2-bnd"
mag_slices2 = [22., 22.5, 23.0, 23.5, 23.75, 23.825]

# XD1-XD2-boundary difference plots
plot_bnd_diff = True
plot_bnd_diff_movie = False
diff_bnd_fname = "XD1-XD2-diff"
diff_bnd_fig_directory = "./bnd_fig_directory/XD1-XD2-diff/"


# dNdm plots
plot_dNdm = True # Plot XD1.
plot_dNdm2 = True # Plot XD2 in addition to XD1 with XD2 as a reference.
dNdm_fname = "dNdm-XD1-XD2-Total"
dNdm_plot_type = "Total" # "Total" if plotting all that are selected, "DESI" if plotting the projection.
dNdm_label1 = "XD1"
dNdm_label2 = "XD2"

# dNdz plots
dz = 0.05 # Redshift binwidth
plot_dNdz = True # Plot XD1.
plot_dNdz2 = True # Plot XD2 in addition to XD1 with XD2 as a reference.
FDR_comparison = True # If True and plot_dNdz2 is False, then dNdz based on XD1 in compared to FDR cut.
dNdz_fname = "dNdz-XD1-XD2"
dNdz_label1 = "XD1"
dNdz_label2 = "XD2"





##############################################################################
# --- XD projection 1 Parameters --- #
# This is a directory where all XD parameters required for projections are
# stored. See README for download instruction.
param_directory = "./XD-parameters/"

# GMM: For the color-color space GMM, use parameters trained on which subset? 
# Field 2, 3, and 4: ""
# Field 2 data only: "-Field2"
# Field 3 or 4 data only: "-Field34"
GMM_which_subset="-Field34"

# dNdm: For the dNdm model, use parameters trained on which subset? 
# Field 2, 3, and 4: ""
# Field 2 data only: "-Field2"
# Field 3 or 4 data only: "-Field34"
dNdm_which_subset="-Field2"

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
f_i=[1., 1., 0., 0.25, 0., .25, 0.]

# Reference points based on which the number density conserving noise are
# calculated.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
gr_ref=0.5
rz_ref=0.5

# Regularizing parameter to be added to the denomitor when calculating FoM.
# Note: Keep the default value 2e-3 unless pathologic behavior boundary occurs,
# in which case it should be raised to a higher value.
reg_r=1e-3

# Grid parameters. A finer grid will slowdown the calculation but may 
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

# GMM: For the color-color space GMM, use parameters trained on which subset? 
# Field 2, 3, and 4: ""
# Field 2 data only: "-Field2"
# Field 3 or 4 data only: "-Field34"
GMM_which_subset2="-Field34"

# dNdm: For the dNdm model, use parameters trained on which subset? 
# Field 2, 3, and 4: ""
# Field 2 data only: "-Field2"
# Field 3 or 4 data only: "-Field34"
dNdm_which_subset2="-Field34"

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
# Note: Keep the default value 2-e3 unless pathologic behavior boundary occurs,
# in which case it should be raised to a higher value.
reg_r2=1e-3

# Grid parameters. A finer grid will slowdown the calculation but may 
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
print("Load DR3-DEEP2 data. g<24.")
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
num_Field2 = oii2.size # Number of Field 2 objects.


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
areas = np.loadtxt("intersection-area-f234")
area_34 = areas[1:].sum()
area_2 = areas[0]
area  = areas.sum()

# Combine all three fields
cn = np.concatenate((cn2,cn3,cn4))
w = np.concatenate((w2, w3, w4))
grz = combine_grz(grz2, grz3, grz4)
grzflux = combine_grz(grzflux2, grzflux3, grzflux4)
grzivar = combine_grz(grzivar2, grzivar3, grzivar4)
d2m = np.concatenate((d2m2, d2m3, d2m4))
redz = np.concatenate((redz2, redz3, redz4))
oii = np.concatenate((oii2, oii3, oii4))
# Giving unmatched objects its proper number.
cn[cn<0] = 7
num_unmatched = (d2m==0).sum()
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# Field 2, 3, and 4 boolean
iF2 = np.zeros(cn.size, dtype=bool)
iF2[:num_Field2] = True
iF34 = np.zeros(cn.size, dtype=bool)
iF34[num_Field2:] = True

# print("Fraction of expected good objects with the rising OII threshold due \n \
# to the increased fiber number and shorter exposure time.")
iGoldSilver = np.logical_or((cn==0), (cn==1))
oii_goldsilver = oii[iGoldSilver]*1e17
w_goldsilver = w[iGoldSilver]

DESI_frac = frac_above_new_oii(oii_goldsilver, w_goldsilver, new_oii_lim(N_tot, 2400))
if two_projections:	
	DESI_frac2 = frac_above_new_oii(oii_goldsilver, w_goldsilver, new_oii_lim(N_tot2, 2400))

print("Completed.\n")


##############################################################################
print("Load DR3-DEEP2 data. r<23p4. For the purpose of comparison with FDR cut.")
# Field 2
set2_FDR = load_fits_table("DECaLS-DR3-DEEP2f2-rlim23p4.fits")
set2_FDR = set2_FDR[reasonable_mask(set2_FDR)] # Applying reasonable mask
grz2_FDR = load_grz(set2_FDR)
cn2_FDR = load_cn(set2_FDR)
w2_FDR = load_weight(set2_FDR)
grzflux2_FDR = load_grz_flux(set2_FDR)
grzivar2_FDR = load_grz_invar(set2_FDR)
d2m2_FDR = load_DEEP2matched(set2_FDR) # DEEP2_matched?
redz2_FDR = load_redz(set2_FDR)
oii2_FDR = load_oii(set2_FDR)
num_Field2_FDR = oii2_FDR.size # Number of Field 2 objects.


# Field 3
set3_FDR = load_fits_table("DECaLS-DR3-DEEP2f3-rlim23p4.fits")
set3_FDR = set3_FDR[reasonable_mask(set3_FDR)] # Applying reasonable mask
grz3_FDR = load_grz(set3_FDR)
cn3_FDR = load_cn(set3_FDR)
w3_FDR = load_weight(set3_FDR)
grzflux3_FDR = load_grz_flux(set3_FDR)
grzivar3_FDR = load_grz_invar(set3_FDR)
d2m3_FDR = load_DEEP2matched(set3_FDR) # DEEP2_matched? 
redz3_FDR = load_redz(set3_FDR)
oii3_FDR = load_oii(set3_FDR)

# Field 4
set4_FDR = load_fits_table("DECaLS-DR3-DEEP2f4-rlim23p4.fits")
set4_FDR = set4_FDR[reasonable_mask(set4_FDR)] # Applying reasonable mask
grz4_FDR = load_grz(set4_FDR)
cn4_FDR = load_cn(set4_FDR)
w4_FDR = load_weight(set4_FDR)
grzflux4_FDR = load_grz_flux(set4_FDR)
grzivar4_FDR = load_grz_invar(set4_FDR)
d2m4_FDR = load_DEEP2matched(set4_FDR) # DEEP2_matched? 
redz4_FDR = load_redz(set4_FDR)
oii4_FDR = load_oii(set4_FDR)


# Combine all three fields
cn_FDR = np.concatenate((cn2_FDR, cn3_FDR, cn4_FDR))
w_FDR = np.concatenate((w2_FDR, w3_FDR, w4_FDR))
grz_FDR = combine_grz(grz2_FDR, grz3_FDR, grz4_FDR)
grzflux_FDR = combine_grz(grzflux2_FDR, grzflux3_FDR, grzflux4_FDR)
grzivar_FDR= combine_grz(grzivar2_FDR, grzivar3_FDR, grzivar4_FDR)
d2m_FDR = np.concatenate((d2m2_FDR, d2m3_FDR, d2m4_FDR))
redz_FDR = np.concatenate((redz2_FDR, redz3_FDR, redz4_FDR))
oii_FDR = np.concatenate((oii2_FDR, oii3_FDR, oii4_FDR))
# Giving unmatched objects its proper number.
cn_FDR[cn_FDR<0] = 7
num_unmatched_FDR = (d2m_FDR==0).sum()
# print("Total number of unmatched objects: %d" % num_unmatched)
# print("In density: %.2f" % (num_unmatched/area.sum()))
# print((cn<0).sum())

# Field 2, 3, and 4 boolean
iF2_FDR = np.zeros(cn_FDR.size, dtype=bool)
iF2_FDR[:num_Field2] = True
iF34_FDR = np.zeros(cn_FDR.size, dtype=bool)
iF34_FDR[num_Field2:] = True

print("Completed.\n")


##############################################################################
print("Compute XD projection 1.")
start = time.time()
grid, last_FoM = XD.generate_XD_selection(param_directory, glim=glim, rlim=rlim, zlim=zlim, \
                          gr_ref=gr_ref, rz_ref=rz_ref, N_tot=N_tot, f_i=f_i, \
                          reg_r=reg_r,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = minmag, \
                          maxmag = maxmag, K_i = K_i, dNdm_type = dNdm_type, \
                          param_tag_GMM = GMM_which_subset, param_tag_dNdm = dNdm_which_subset)
print("Time taken: %.2f seconds" % (time.time()-start))
print("Computed last FoM based on the grid: %.3f"%last_FoM)
print("Completed.\n")

if two_projections:
	print("Compute XD projection 2.")
	start = time.time()
	grid2, last_FoM2 = XD.generate_XD_selection(param_directory2, glim=glim2, rlim=rlim2, zlim=zlim2, \
	                          gr_ref=gr_ref2, rz_ref=rz_ref2, N_tot=N_tot2, f_i=f_i2, \
	                          reg_r=reg_r2,zaxis="g", w_cc = w_cc2, w_mag = w_mag2, minmag = minmag2, \
	                          maxmag = maxmag2, K_i = K_i2, dNdm_type = dNdm_type2, \
	                          param_tag_GMM = GMM_which_subset2, param_tag_dNdm = dNdm_which_subset2)
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
                       gmin = gmin, gmax = gmax, K_i = K_i, dNdm_type = dNdm_type, \
                      param_tag_GMM = GMM_which_subset, param_tag_dNdm = dNdm_which_subset)
print("Completed.\n")

if two_projections:
	print("Apply XD selection 2 to DEEP2-DECaLS data.")
	iXD2, FoM2 = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM2, param_directory2, \
	                        glim=glim2, rlim=rlim2, zlim=zlim2, gr_ref=gr_ref2,\
	                       rz_ref=rz_ref2, reg_r=reg_r2/(w_cc2**2 * w_mag2), f_i=f_i2,\
	                       gmin = gmin2, gmax = gmax2, K_i = K_i2, dNdm_type = dNdm_type2, 
                          param_tag_GMM = GMM_which_subset2, param_tag_dNdm = dNdm_which_subset2)
	print("Completed.\n")


##############################################################################
print("Compute FDR cut.")
iFDR = FDR_cut(grz_FDR)
print("Completed.\n")


##############################################################################
print("Print projection results in tabular form.")
print(" & ".join(["Cut", "Type", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "FoM"]) + "\\\\ \hline")

# FDR
return_format = ["FDR", "F234", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn_FDR[iFDR], w_FDR[iFDR], area,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# FDR - Field 3 and 4
return_format = ["FDR", "F34", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn_FDR[iFDR & iF34_FDR], w_FDR[iFDR & iF34_FDR], area_34,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# FDR - Field 2
return_format = ["FDR", "F2", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn_FDR[iFDR & iF2_FDR], w_FDR[iFDR & iF2_FDR], area_2,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD cut
return_format = ["XD1", "F234", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD], w[iXD], area,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD cut - Field 3 and 4
return_format = ["XD1", "F34", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD & iF34], w[iXD & iF34], area_34,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD cut - Field 2
return_format = ["XD1", "F2", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD & iF2], w[iXD & iF2], area_2,rwd="D", num_classes=8, \
     return_format = return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0. ,0.]))

# XD projection
return_format = ["XD1", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
print(class_breakdown_cut_grid(grid, return_format, class_eff = [gold_eff*DESI_frac, gold_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0.]))

if two_projections:
	# XD cut
	return_format = ["XD2", "F234", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM2),  "\\\\ \hline"]
	print(class_breakdown_cut(cn[iXD2], w[iXD2], area,rwd="D", num_classes=8, \
	     return_format = return_format, class_eff = [gold_eff2*DESI_frac2, gold_eff2*DESI_frac2, 0.0, NoOII_eff2*DESI_frac2, 0., NoZ_eff2*DESI_frac2, 0., 0.]))

	# XD cut - Field 3 and 4
	return_format = ["XD2", "F34", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM2),  "\\\\ \hline"]
	print(class_breakdown_cut(cn[iXD2 & iF34], w[iXD2 & iF34], area_34,rwd="D", num_classes=8, \
	     return_format = return_format, class_eff = [gold_eff2*DESI_frac2, gold_eff2*DESI_frac2, 0.0, NoOII_eff2*DESI_frac2, 0., NoZ_eff2*DESI_frac2, 0., 0.]))

	# XD cut - Field 2
	return_format = ["XD2", "F2", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
	print(class_breakdown_cut(cn[iXD2 & iF2], w[iXD2 & iF2], area_2,rwd="D", num_classes=8, \
	     return_format = return_format, class_eff = [gold_eff2*DESI_frac2, gold_eff2*DESI_frac2, 0.0, NoOII_eff2*DESI_frac2, 0., NoZ_eff2*DESI_frac2, 0., 0.]))

	# XD projection
	return_format = ["XD2", "Proj.", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
	      "DESI", "Total", "Eff", str("%.3f"%last_FoM2),  "\\\\ \hline"]
	print(class_breakdown_cut_grid(grid2, return_format, class_eff = [gold_eff2*DESI_frac2, gold_eff2*DESI_frac2, 0.0, NoOII_eff2*DESI_frac2, 0., NoZ_eff2*DESI_frac2, 0.]))

print("Completed.\n")


##############################################################################
# Make boundary plots.
if plot_bnd:
	print("XD1: Create plots of boundary at various magnitudes.")
	for m in mag_slices:
	    print("Mag %.3f"%m)
	    XD.plot_slice(grid, m, bnd_fig_directory, bnd_fname)
	print("Completed.\n")

if two_projections and plot_bnd2:
	print("XD2: Create plots of boundary at various magnitudes.")
	for m in mag_slices2:
	    print("Mag %.3f"%m)
	    XD.plot_slice(grid2, m, bnd_fig_directory2, bnd_fname2)
	print("Completed.\n")


##############################################################################
# Many boundary plots for generating a movie.
if plot_bnd_movie:
	print("XD1: Create many plots of boundary at various magnitudes to generate a movie.")
	dm = w_mag
	for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
	    print("Index %d, Slice %.3f" % (i,m))
	    XD.plot_slice(grid, m, bnd_fig_directory, bnd_fname, movie_tag=i)   

	print("Completed.\n")

	print("Command for creating a movie.:\n \
	    ffmpeg -r 6 -start_number 0 -i %s-mag0-%%d.png -vcodec mpeg4 -y %s-movie.mp4"%(bnd_fname, bnd_fname))

if two_projections and plot_bnd_movie2:
	print("XD2: Create many plots of boundary at various magnitudes to generate a movie.")
	dm = w_mag2
	for i,m in enumerate(np.arange(21.5,24+0.9*w_mag2, w_mag2)):
	    print("Index %d, Slice %.3f" % (i,m))
	    XD.plot_slice(grid2, m, bnd_fig_directory2, bnd_fname2, movie_tag=i)   

	print("Completed.\n")

	print("Command for creating a movie.:\n \
	    ffmpeg -r 6 -start_number 0 -i %s-mag0-%%d.png -vcodec mpeg4 -y %s-movie.mp4"%(bnd_fname2, bnd_fname2))		




##############################################################################
# Make comparison boundary plots.
if  two_projections and plot_bnd_diff:
	print("Plot figures that compare boundaries XD1 and XD2 (reference).")
	if (np.abs(w_mag-w_mag2)<1e-6) and (np.abs(w_cc-w_cc2)<1e-6):
		for m in mag_slices:
		    print("Mag %.3f"%m)
		    XD.plot_slice_compare(grid, grid2, m, diff_bnd_fig_directory, diff_bnd_fname)
		print("Completed.\n")
	else:		print("The grid dimensions must be the same for this comparison.")



##############################################################################
# Many comparison boundary plots for generating a movie.
if two_projections and plot_bnd_diff_movie:
	print("Generate a movie comparing boundaries XD1 and XD2 (reference).")
	if (np.abs(w_mag-w_mag2)<1e-6) and (np.abs(w_cc-w_cc2)<1e-6):
		dm = w_mag
		for i,m in enumerate(np.arange(21.5,24+0.9*w_mag, w_mag)):
			print("Index %d, Slice %.3f" % (i,m))
			XD.plot_slice_compare(grid, grid2, m, diff_bnd_fig_directory, diff_bnd_fname, movie_tag=i)

		print("Completed.\n")

		print("Command for creating a movie.:\n \
		    ffmpeg -r 6 -start_number 0 -i %s-mag0-%%d.png -vcodec mpeg4 -y %s-movie.mp4"%(diff_bnd_fname, diff_bnd_fname))
	else:
		print("The grid dimensions must be the same for this comparison.")



# ##############################################################################
if plot_dNdm:
	print("Make dNdm plots.")	
	if two_projections and plot_dNdm2:
		# For both
		XD.plot_dNdm_XD(grid, grid2, fname=dNdm_fname, plot_type=dNdm_plot_type, label1 =dNdm_label1, label2 = dNdm_label2, \
			class_eff =  [gold_eff*DESI_frac, silver_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0.],\
			class_eff2 =  [gold_eff2*DESI_frac2, silver_eff2*DESI_frac2, 0.0, NoOII_eff2*DESI_frac2, 0., NoZ_eff2*DESI_frac2, 0.])
	else:
		XD.plot_dNdm_XD(grid, fname=dNdm_fname, plot_type=dNdm_plot_type, label1 = dNdm_label1, \
			class_eff =  [gold_eff*DESI_frac, silver_eff*DESI_frac, 0.0, NoOII_eff*DESI_frac, 0., NoZ_eff*DESI_frac, 0.])
	print("Completed.\n")



##############################################################################
if plot_dNdz:
	print("Plot n(z) for XD projections.")
	if two_projections and plot_dNdz2:
		# For both - Field 2, 3, and 4
		plot_dNdz_selection(cn, w, iXD, redz, area, dz=dz,\
			iselect2=iXD2, plot_total=False, fname=dNdz_fname, color1="red", color2="black", \
			label1=dNdz_label1, label2=dNdz_label2, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff2*DESI_frac2, silver_eff2 = silver_eff2*DESI_frac2, \
			NoOII_eff2 = DESI_frac2*NoOII_eff2, NoZ_eff2 = DESI_frac2*NoZ_eff2)

		# For both - Field 3 and 4
		plot_dNdz_selection(cn[iF34], w[iF34], iXD[iF34], redz[iF34], area_34, dz=dz,\
			iselect2=iXD2[iF34], plot_total=False, fname=dNdz_fname+"-Field34", color1="red", color2="black", \
			label1=dNdz_label1, label2=dNdz_label2, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff2*DESI_frac2, silver_eff2 = silver_eff2*DESI_frac2, \
			NoOII_eff2 = DESI_frac2*NoOII_eff2, NoZ_eff2 = DESI_frac2*NoZ_eff2)

		# For both - Field 2
		plot_dNdz_selection(cn[iF2], w[iF2], iXD[iF2], redz[iF2], area_2, dz=dz,\
			iselect2=iXD2[iF2], plot_total=False, fname=dNdz_fname+"-Field2", color1="red", color2="black", \
			label1=dNdz_label1, label2=dNdz_label2, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff2*DESI_frac2, silver_eff2 = silver_eff2*DESI_frac2, \
			NoOII_eff2 = DESI_frac2*NoOII_eff2, NoZ_eff2 = DESI_frac2*NoZ_eff2)
		print("Completed.\n")
	elif FDR_comparison:
		# For both - Field 2, 3, and 4
		plot_dNdz_selection(cn, w, iXD, redz, area, dz=dz,\
			iselect2=iFDR, plot_total=False, fname=dNdz_fname, color1="red", color2="black", \
			label1=dNdz_label1, label2="FDR", gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff*DESI_frac, silver_eff2 = silver_eff*DESI_frac, \
			NoOII_eff2 = DESI_frac*NoOII_eff, NoZ_eff2 = DESI_frac*NoZ_eff)

		# For both - Field 3 and 4
		plot_dNdz_selection(cn[iF34], w[iF34], iXD[iF34], redz[iF34], area_34, dz=dz,\
			iselect2=iFDR[iF34], plot_total=False, fname=dNdz_fname+"-Field34", color1="red", color2="black", \
			label1=dNdz_label1, label2="FDR", gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff*DESI_frac, silver_eff2 = silver_eff*DESI_frac, \
			NoOII_eff2 = DESI_frac*NoOII_eff, NoZ_eff2 = DESI_frac*NoZ_eff)

		# For both - Field 2
		plot_dNdz_selection(cn[iF2], w[iF2], iXD[iF2], redz[iF2], area_2, dz=dz,\
			iselect2=iFDR[iF2], plot_total=False, fname=dNdz_fname+"-Field2", color1="red", color2="black", \
			label1=dNdz_label1, label2="FDR", gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff, \
			gold_eff2 = gold_eff*DESI_frac, silver_eff2 = silver_eff*DESI_frac, \
			NoOII_eff2 = DESI_frac*NoOII_eff, NoZ_eff2 = DESI_frac*NoZ_eff)
		print("Completed.\n")		
	else:
		# For single - Field 2, 3, and 4
		plot_dNdz_selection(cn, w, iXD, redz, area, dz=dz,\
			iselect2=None, plot_total=False, fname=dNdz_fname, color1="black", \
			label1=dNdz_label1, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff)

		# For single - Field 3 and 4
		plot_dNdz_selection(cn[iF34], w[iF34], iXD[iF34], redz[iF34], area_34, dz=dz,\
			iselect2=None, plot_total=False, fname=dNdz_fname+"-Field34", color1="black", \
			label1=dNdz_label1, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff)

		# For single - Field 2
		plot_dNdz_selection(cn[iF2], w[iF2], iXD[iF2], redz[iF2], area_2, dz=dz,\
			iselect2=None, plot_total=False, fname=dNdz_fname+"-Field2", color1="black", \
			label1=dNdz_label1, gold_eff = gold_eff*DESI_frac, silver_eff = silver_eff*DESI_frac, \
			NoOII_eff = DESI_frac*NoOII_eff, NoZ_eff = DESI_frac*NoZ_eff)					
		print("Completed.\n")		