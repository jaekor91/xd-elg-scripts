# Import packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.wcs import WCS
import numpy.lib.recfunctions as rec
from xd_elg_utils import *

# Constants
large_random_constant = -999119283571
deg2arcsec=3600

# Data directory (set by the user)
data_directory = "./"

##############################################################################
print("1. Load DEEP2 pcats files for Field 2, 3, and 4.")
# fp = 21, 22
pcat21 = fits.open(data_directory+"pcat_ext.21.fits")
pcat22 = fits.open(data_directory+"pcat_ext.22.fits")
pcat2 = np.hstack((pcat21[1].data, pcat22[1].data))
pcat21.close() # Closing files
pcat22.close()

# fp = 31, 32, 33
pcat31 = fits.open(data_directory+"pcat_ext.31.fits")
pcat32 = fits.open(data_directory+"pcat_ext.32.fits")
pcat33 = fits.open(data_directory+"pcat_ext.33.fits")
pcat3 = np.hstack((pcat31[1].data, pcat32[1].data,pcat33[1].data))
pcat31.close() # Closing files
pcat32.close()
pcat33.close()

# field4
print("Field 4. Expected warning.")
pcat4= fits.open(data_directory+"deep2-f4-photo-newman.fits")[1].data

print("Completed.\n")


##############################################################################
print("2. Impose BADFLAG==0 mask")
# Field 2
pcat2_good = pcat2[pcat2["BADFLAG"]==0]

# Field 3
pcat3_good = pcat3[pcat3["BADFLAG"]==0]

# Field 4
pcat4_good = pcat4

print("Completed.\n")


##############################################################################
print("3. Impose window function mask")
# Field 2
idx = np.logical_or(window_mask(pcat2_good["RA_DEEP"], pcat2_good["DEC_DEEP"], "windowf.21.fits"), window_mask(pcat2_good["RA_DEEP"], pcat2_good["DEC_DEEP"], "windowf.22.fits"))
pcat2_trimmed = pcat2_good[idx]

# Field 3
idx = np.logical_or.reduce((window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], "windowf.31.fits"), window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], "windowf.32.fits"),window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], "windowf.33.fits")))
pcat3_trimmed = pcat3_good[idx]

# Field 4
idx = np.logical_or(window_mask(pcat4_good["RA"], pcat4_good["DEC"], "windowf.41.fits"), window_mask(pcat4_good["RA"], pcat4_good["DEC"], "windowf.42.fits"))
pcat4_trimmed = np.copy(pcat4_good[idx])

print("Completed.\n")


##############################################################################
print("4. Save the trimmed DEEP2 photometric catalogs")
# Field 2
cols = fits.ColDefs(pcat2_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f2-photo-trimmed.fits', clobber=True)

# Field 3
cols = fits.ColDefs(pcat3_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f3-photo-trimmed.fits', clobber=True)

# Field 4
cols = fits.ColDefs(pcat4_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f4-photo-trimmed.fits', clobber=True)

print("Completed.\n")


##############################################################################
print("5. Estimate the spectroscopic area using window functions")
# Field 2
area_f2 = est_spec_area("windowf.21.fits")+est_spec_area("windowf.22.fits")
print("Field 2 area: %.3f" % area_f2)

# Field 3
area_f3 = est_spec_area("windowf.31.fits")+est_spec_area("windowf.32.fits")+est_spec_area("windowf.33.fits")
print("Field 3 area: %.3f" % area_f3)

# Field 4
area_f4 = est_spec_area("windowf.41.fits")+est_spec_area("windowf.42.fits")
print("Field 4 area: %.3f" % area_f4)

# Total
print("Total DEEP2 spectroscopic area: %.3f" % (area_f2+area_f3+area_f4))

print("Completed.\n")


##############################################################################
print("6. Load other catalogs.")
print("color-selection.txt: Catalog that provides DEEP2 BRI color selection information.\n \
	 Contains ra, dec, and color selection flag. object ID, RA, dec, whether the object \n \
	 would have been targeted if in EGS (1=yes, 0=no), and whether it would have been \n \
	 targeted in a non-EGS field. Provided by Jeff Newman")
DEEP2color = ascii.read("color-selection.txt")
DEEP2color_OBJNO = np.asarray(DEEP2color["col1"])
DEEP2color_ra = np.asarray(DEEP2color["col2"])
DEEP2color_dec = np.asarray(DEEP2color["col3"])
# DEEP2color_EGS=np.asarray(DEEP2color["col4"])
DEEP2color_BRI=np.asarray(DEEP2color["col5"])
print("Completed.\n")

print("selection-prob.fits: Catalog that provides selection weight information, OBJNO\n\
		 and P_onmask. Provided by Jeff Newman.")
DEEP2weight = fits.open("selection-prob.fits")
DEEP2weight_OBJNO = DEEP2weight[1].data["OBJNO"]
DEEP2weight_weight = DEEP2weight[1].data["P_ONMASK"]
DEEP2weight.close()
print("Completed.\n")


print("deep2-f**-redz-oii.fits: DEEP2 redshift catalogs that John Moustakas provided.\n \
	Extract OBJNO, RA, DEC, OII_3727, OII_3727_ERR, ZHELIO, ZHELIO_ERR, ZQUALITY.\n \
	Note 1: oii 3727, oii 3727 err, integrated [OII] flux and uncertainty. Note\n \
	that the uncertainty equals -2.0 if the line could not be measured.\n \
	Note 1b: Negative errors have the following meaning\n \
    \t-1.0 = line not detected with amplitude S/N > 1.5. Upper limit calculated.\n \
    \t-2.0 = line not measured (not in spectral range)\n \
	Note 2: For ZQUALITY values, see http://deep.ps.uci.edu/DR4/zquality.html.")
# Field 2
f2_objno, f2_ra, f2_dec, f2_oii, f2_oii_err, f2_z, f2_z_err, f2_zquality, f2_weight = import_zcat("deep2-f2-redz-oii.fits")

# Field 3
f3_objno, f3_ra, f3_dec, f3_oii, f3_oii_err, f3_z, f3_z_err, f3_zquality, f3_weight = import_zcat("deep2-f3-redz-oii.fits")

# Field 4
f4_objno, f4_ra, f4_dec, f4_oii, f4_oii_err, f4_z, f4_z_err, f4_zquality, f4_weight = import_zcat("deep2-f4-redz-oii.fits")
print("Completed.\n")



##############################################################################
print("7. Append additional columns to the photometric catalogs from others.")
print("7a. Append redshift catalogs.")
col_name_list = ["OBJNO_zcat", "RA_zcat", "DEC_zcat", "OII_3727","OII_3727_ERR", "RED_Z", "Z_ERR", "ZQUALITY_JOHN", "TARG_WEIGHT"]
# Field 2
pcat2 = fits.open("deep2-f2-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat2["OBJNO"], f2_objno)
new_col_list = [f2_objno, f2_ra, f2_dec, f2_oii, f2_oii_err, f2_z, f2_z_err, f2_zquality, f2_weight]
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx1, idx2)
    
# Field 3
pcat3 = fits.open("deep2-f3-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat3["OBJNO"], f3_objno)
new_col_list = [f3_objno, f3_ra, f3_dec, f3_oii, f3_oii_err, f3_z, f3_z_err, f3_zquality, f3_weight]
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx1, idx2)

# Field 4
del pcat4
col_name_list = ["OBJNO_zcat", "RA_zcat", "DEC_zcat", "OII_3727","OII_3727_ERR", "RED_Z", "Z_ERR", "ZQUALITY", "TARG_WEIGHT"]
pcat4 = fits.open("deep2-f4-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat4["OBJNO"], f4_objno)
new_col_list = [f4_objno, f4_ra, f4_dec, f4_oii, f4_oii_err, f4_z, f4_z_err, f4_zquality, f4_weight]
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx1, idx2)

print("f2: # in zcat minus # in pcat matched %d" % (f2_objno.size-(pcat2["RED_Z"]>-1000).sum()))
print("f3: # in zcat minus # in pcat matched %d" % (f3_objno.size-(pcat3["RED_Z"]>-1000).sum()))
print("f4: # in zcat minus # in pcat matched %d" % (f4_objno.size-(pcat4["RED_Z"]>-1000).sum()))    
print("The number of overlapping objects are smaller because certain\n \
	spectroscopic areas were masked out in previous steps (the area\n \
	estimates above are compatible). I ignore the small number of object\n \
	loss. The object loss ")
print("Completed.\n")


print("7b. Append probability information.")
col_name_list = ["OBJNO_prob", "weight"]
new_col_list = [DEEP2weight_OBJNO, DEEP2weight_weight]
# Field 2
idx1, idx2 = match_objno(pcat2["OBJNO"], DEEP2weight_OBJNO)
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx1, idx2)

# Field 3
idx1, idx2 = match_objno(pcat3["OBJNO"], DEEP2weight_OBJNO)
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx1, idx2)    

# Field 4
idx1, idx2 = match_objno(pcat4["OBJNO"], DEEP2weight_OBJNO)
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx1, idx2)
print("Completed.\n")


print("7c. Append color-selection information.")
# List of new columns to be appended
col_name_list = ["OBJNO_color", "RA_color", "DEC_color", "BRI_cut"]
new_col_list = [DEEP2color_OBJNO, DEEP2color_ra, DEEP2color_dec, DEEP2color_BRI]
# Field 2
idx1, idx2 = match_objno(pcat2["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx1, idx2)
    
# Field 3
idx1, idx2 = match_objno(pcat3["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx1, idx2)    

# Field 4
idx1, idx2 = match_objno(pcat4["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx1, idx2)
print("f2: # in pcat minus # in pcat matched %d" % (pcat2.shape[0]-(pcat2["BRI_cut"]>-1000).sum()))
print("f3: # in pcat minus # in pcat matched %d" % (pcat3.shape[0]-(pcat3["BRI_cut"]>-1000).sum()))
print("f4: # in pcat minus # in pcat matched %d" % (pcat4.shape[0]-(pcat4["BRI_cut"]>-1000).sum()))    
print("The last 251 objects will be classified as DEEP2 BRI color rejected objects.")
print("Completed.\n")



##############################################################################
print("8. Append the class column based on the information above.")
col_name_list = ["cn"]
# Field 2
new_col_list = [generate_class_col(pcat2)]
idx = range(pcat2.shape[0])
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx, idx)

    
# Field 3
new_col_list = [generate_class_col(pcat3)]
idx = range(pcat3.shape[0])
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx, idx)
      

# Field 4
new_col_list = [generate_class_col(pcat4)]
idx = range(pcat4.shape[0])
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx, idx)

print("Category counts 0 through 8")
print(np.bincount(pcat2["cn"].astype(int)))
print(np.bincount(pcat3["cn"].astype(int)))
print(np.bincount(pcat4["cn"].astype(int)))
print("Completed.\n")


##############################################################################
print("9. Save the resulting catalog.")
# Field 2
cols = fits.ColDefs(pcat2)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f2-photo-redz-oii.fits', clobber=True)

# Field 3
cols = fits.ColDefs(pcat3)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f3-photo-redz-oii.fits', clobber=True)

# Field 4
cols = fits.ColDefs(pcat4)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('deep2-f4-photo-redz-oii.fits', clobber=True)
print("Completed.\n")


##############################################################################
print("10. Print the class breakdown in latex format.")
table_header = generate_table_header()
print(table_header)
# Raw number
class_breakdown([2, 3, 4],[pcat2["cn"],pcat3["cn"],pcat4["cn"]],[pcat2["TARG_WEIGHT"],pcat3["TARG_WEIGHT"],pcat4["TARG_WEIGHT"]], [area_f2,area_f3,area_f4],rwd="R")
# Weighted number
class_breakdown([2, 3, 4],[pcat2["cn"],pcat3["cn"],pcat4["cn"]],[pcat2["TARG_WEIGHT"],pcat3["TARG_WEIGHT"],pcat4["TARG_WEIGHT"]], [area_f2,area_f3,area_f4],rwd="W")
# Density (weighted divided by area)
class_breakdown([2, 3, 4],[pcat2["cn"],pcat3["cn"],pcat4["cn"]],[pcat2["TARG_WEIGHT"],pcat3["TARG_WEIGHT"],pcat4["TARG_WEIGHT"]], [area_f2,area_f3,area_f4],rwd="D")
print("Note: D2unboserved should be equal to zero for weighted and density cases.")
print("Completed.\n")

