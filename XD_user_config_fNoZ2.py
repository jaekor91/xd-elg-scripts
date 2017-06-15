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
two_projections = True


##############################################################################
# --- Outputs --- #
# XD1 for projection based on first set and XD2 corresponding to the second set

# XD1: Boundary plots 
plot_bnd = True
plot_bnd_movie = False # Generate many slices for a movie.
bnd_fig_directory = "./bnd_fig_directory/fNoZ2/"
bnd_fname = "XD-fNoZ2"
mag_slices = [22., 22.5, 23.0, 23.5, 23.75, 23.825, 23.975]

# XD2: Boundary plots
plot_bnd2 = False
plot_bnd_movie2 = False # Generate many slices for a movie.
bnd_fig_directory2 = "./bnd_fig_directory/XD2-bnd/"
bnd_fname2 = "XD2-bnd"
mag_slices2 = [22., 22.5, 23.0, 23.5, 23.75, 23.825, 23.975]

# XD1-XD2-boundary difference plots
plot_bnd_diff = True
plot_bnd_diff_movie = False
diff_bnd_fname = "fNoZ2-fid-diff"
diff_bnd_fig_directory = "./bnd_fig_directory/fNoZ2-fid-diff/"


# dNdm plots
plot_dNdm = True # Plot XD1.
plot_dNdm2 = True # Plot XD2 in addition to XD1 with XD2 as a reference.
dNdm_fname = "dNdm-fNoZ2-fiducial-Total"
dNdm_plot_type = "Total" # "Total" if plotting all that are selected, "DESI" if plotting the projection.
dNdm_label1 = "fNoZ = 2"
dNdm_label2 = "Fid."

# dNdz plots
dz = 0.05 # Redshift binwidth
plot_dNdz = True # Plot XD1.
plot_dNdz2 = True # Plot XD2 in addition to XD1 with XD2 as a reference.
FDR_comparison = False # If True and plot_dNdz2 is False, then dNdz based on XD1 in compared to FDR cut.
dNdz_fname = "dNdz-fNoZ2-fiducial"
dNdz_label1 = "fNoZ = 2"
dNdz_label2 = "Fid."





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
f_i=[1., 1., 0., 0.25, 0., 2., 0.]

# Reference points based on which the number density conserving noise are
# calculated.
# Note: Do not change these parameters unless the user is sure of what
# is being done.
gr_ref=0.5
rz_ref=0.5

# Regularizing parameter to be added to the denomitor when calculating FoM.
# Note: Keep the default value 2e-3 unless pathologic behavior boundary occurs,
# in which case it should be raised to a higher value.
reg_r=2e-3

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
dNdm_which_subset2="-Field2"

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