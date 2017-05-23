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


##############################################################################
param_directory = "./"
f_i = [1., 1., 0., 0.25, 0., 0.25, 0.]
param_directory = "./"
w_mag = 0.05/2.
w_cc = 0.025/2.
N_tot = 2400
glim = 23.8
rlim = 23.4
zlim = 22.4
reg_r = 5e-4 
g,r,z = grz
givar, rivar, zivar = grzivar
gflux, rflux, zflux = grzflux


##############################################################################
# FDR Cut
iFDR = FDR_cut(grz)
print(" & ".join(["Cut", "Type", "glim", "rlim", "zlim", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "FoM"]) + "\\\\ \hline")
# FDR
return_format = ["FDR", "Avg.", "--", "--", "--", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", "--",  "\\\\ \hline"]
print(class_breakdown_cut(cn[iFDR], w[iFDR], area,rwd="D", num_classes=8, \
     return_format = return_format))

##############################################################################
# Compute XD projection based on fiducial set of parameters.
grid_fiducial, last_FoM_fiducial = XD.generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
                          reg_r=reg_r,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])

iXD_fiducial, FoM_fiducial = XD.apply_XD_globalerror([g, r, z, givar, rivar, zivar, gflux, rflux, zflux], last_FoM_fiducial, param_directory, \
            glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5, reg_r=reg_r/(w_cc**2 * w_mag), f_i=f_i,\
                       gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])

# XD projection
return_format = ["XD", "Proj.", str("%.2f"%glim), str("%.2f"%rlim), str("%.2f"%zlim),"Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM_fiducial),  "\\\\ \hline"]
print(class_breakdown_cut_grid(grid_fiducial, return_format))

# XD cut
return_format = ["XD", "Avg.","--", "--","--", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched", \
      "DESI", "Total", "Eff", str("%.3f"%last_FoM_fiducial),  "\\\\ \hline"]
print(class_breakdown_cut(cn[iXD_fiducial], w[iXD_fiducial], area,rwd="D", num_classes=8, \
     return_format = return_format))



  # ##############################################################################
# Compute XD density projection based on non-fiducial set of depth parameters 
# but make selection assuming fiducial depths.
reg_r = 5e-4

dm_var = 0.125
dm_varX2 = dm_var*2

dm_list = [-dm_varX2, -dm_var, dm_var, dm_varX2]

for dm in dm_list:
	for k in range(3):
		glim_var = glim
		rlim_var = rlim
		zlim_var = zlim
		if k == 0:
			glim_var += dm
		elif k == 1:
			rlim_var += dm
		elif k == 2:
			zlim_var += dm
		grid, last_FoM, last_FoM_var = XD.generate_XD_selection_var(param_directory, glim=glim, rlim=rlim, zlim=zlim, \
									glim_var=glim_var, rlim_var=rlim_var, zlim_var=zlim_var, \
									gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=f_i, \
									reg_r=reg_r,zaxis="g", w_cc = w_cc, w_mag = w_mag, minmag = 21.5+w_mag/2., \
									maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1])

		# XD projection - Selection based on "true" depth
		return_format = ["XD", "Adap.", str("%.2f"%glim_var), str("%.2f"%rlim_var), str("%.2f"%zlim_var),"Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
		      "DESI", "Total", "Eff", str("%.3f"%last_FoM_var),  "\\\\ \hline"]
		print(class_breakdown_cut_grid(grid, return_format, selection="select_var"))

		# XD projection - Selection based on assumed depth
		return_format = ["XD", "Fid.", "--", "--","--","Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "--", \
		      "DESI", "Total", "Eff", str("%.3f"%last_FoM),  "\\\\ \hline"]
		print(class_breakdown_cut_grid(grid, return_format))


		##############################################################################
		# Make dNdm plots.
		# Total 
		fname = "dNdm-XD-glimvar%d-rlimvar%d-zlimvar%d-fiducial-dNdm-Total"%(glim_var*100,rlim_var*100,zlim_var*100)
		XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, plot_type="Total", label1="Fid. Glb.", label2 ="Fid. Orig.", label3="Adap.", glim2=glim_var, rlim2=rlim_var, zlim2=zlim_var)

		# DESI
		fname = "dNdm-XD-glimvar%d-rlimvar%d-zlimvar%d-fiducial-dNdm-DESI"%(glim_var*100,rlim_var*100,zlim_var*100)
		XD.plot_dNdm_XD(grid, grid_fiducial, fname=fname, plot_type="DESI", label1="Fid. Glb.", label2 ="Fid. Orig.", label3="Adap.", glim2=glim_var, rlim2=rlim_var, zlim2=zlim_var)



