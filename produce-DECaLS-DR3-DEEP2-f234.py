
# Load modules
import numpy as np
from astropy.io import fits
import numpy.lib.recfunctions as rec
from xd_elg_utils import *

large_random_constant = -9991191
deg2arcsec=3600


##############################################################################
print("1. Load processed DEEP2 and DR3 catalogs")
print("DEEP2 catalogs. deep2-f**-photo-redz-oii.fits")
deep2f2 = load_fits_table("deep2-f2-photo-redz-oii.fits")
deep2f3 = load_fits_table("deep2-f3-photo-redz-oii.fits")
deep2f4 = load_fits_table("deep2-f4-photo-redz-oii.fits")

# DR3 catalog and load stark mask immediately.
print("DECaLS catalogs. DECaLS-DR3-Tractor-DEEP2f2**.fits\n\
	Apply Tycho-2 stellar mask.")
dr3f2 = load_fits_table("DECaLS-DR3-Tractor-DEEP2f2.fits")
dr3f2 = apply_star_mask(dr3f2)
dr3f3 = load_fits_table("DECaLS-DR3-Tractor-DEEP2f3.fits")
dr3f3 = apply_star_mask(dr3f3)
dr3f4 = load_fits_table("DECaLS-DR3-Tractor-DEEP2f4.fits")
dr3f4 = apply_star_mask(dr3f4)

print("Load DR3 grz fluxes and apply positivity condition.")
# Flux positivity mask
grz2flux = load_grz_flux(dr3f2)
grz3flux = load_grz_flux(dr3f3)
grz4flux = load_grz_flux(dr3f4)
ibool2 = is_grzflux_pos(grz2flux)
ibool3 = is_grzflux_pos(grz3flux)
ibool4 = is_grzflux_pos(grz4flux)
dr3f2 = dr3f2[ibool2]
dr3f3 = dr3f3[ibool3]
dr3f4 = dr3f4[ibool4]

print("Load grz magnitudes.")
# Loading mag and flux
grz2 = load_grz(dr3f2)
grz3 = load_grz(dr3f3)
grz4 = load_grz(dr3f4)

print("Completed.\n")

##############################################################################
print("2. Cross-matching DR3 objects with g<24.0 and DEEP2 objects within each field.\n\
    Check astrometric difference.")

glim = 24.0
print("Field 2")
dr3f2_glim = dr3f2[grz2[0]<glim]
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f2_glim["RA"],dr3f2_glim["DEC"],deep2f2["RA_DEEP"], deep2f2["DEC_DEEP"], tol=1./3600.)
ra_med_diff_f2, dec_med_diff_f2 = check_astrometry(dr3f2_glim["RA"][idx1],dr3f2_glim["DEC"][idx1],deep2f2["RA_DEEP"][idx2], deep2f2["DEC_DEEP"][idx2],pt_size=0.1)
print("# of matches %d"%idx1.size)
print("ra, dec median differences in arcsec: %.3f, %.3f\n" %(ra_med_diff_f2*deg2arcsec, dec_med_diff_f2*deg2arcsec))

print("Field 3")
dr3f3_glim = dr3f3[grz3[0]<glim]
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f3_glim["RA"],dr3f3_glim["DEC"],deep2f3["RA_DEEP"], deep2f3["DEC_DEEP"], tol=1./3600.)
ra_med_diff_f3, dec_med_diff_f3 = check_astrometry(dr3f3_glim["RA"][idx1],dr3f3_glim["DEC"][idx1],deep2f3["RA_DEEP"][idx2], deep2f3["DEC_DEEP"][idx2],pt_size=0.1)
print("# of matches %d"%idx1.size)
print("ra, dec median differences in arcsec: %.3f, %.3f\n" %(ra_med_diff_f3*deg2arcsec, dec_med_diff_f3*deg2arcsec))

print("Field 4")
dr3f4_glim = dr3f4[grz4[0]<glim]
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f4_glim["RA"],dr3f4_glim["DEC"],deep2f4["RA"], deep2f4["DEC"], tol=1./3600.)
ra_med_diff_f4, dec_med_diff_f4 = check_astrometry(dr3f4_glim["RA"][idx1],dr3f4_glim["DEC"][idx1],deep2f4["RA"][idx2], deep2f4["DEC"][idx2],pt_size=0.1)
print("# of matches %d"%idx1.size)
print("ra, dec median differences in arcsec: %.3f, %.3f\n" %(ra_med_diff_f4*deg2arcsec, dec_med_diff_f4*deg2arcsec))

print("Completed.\n")


##############################################################################
print("3. Crossmatch the catalogs taking into account astrometric differences.\n\
	Save 1) DECaLS appended with DEEP2 and 2) Unmatched DEEP2.")

print("The following DEEP2 fields were appended to DECaLS: [('OBJNO', '>i4'), \n\
	 ('MAGB', '>f4'), ('MAGR', '>f4'), ('MAGI', '>f4'),\n\
	 ('MAGBERR', '>f4'), ('MAGRERR', '>f4'), ('MAGIERR', '>f4'), ('BADFLAG', 'u1'),\n\
	 ('OII_3727', '>f8'), ('OII_3727_ERR', '>f8'), ('RED_Z', '>f8'), ('Z_ERR', '>f8'),\n\
	 ('ZQUALITY', '>f8'), ('TARG_WEIGHT', '>f8'), ('weight', '>f8'), ('BRI_cut', '>f8'),\n\
	 ('cn', '>f8')]\n \
	 Note that the appended column names are the same for Fields 2, 3, and 4.")


print("Field 2")
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f2_glim["RA"]+ra_med_diff_f2,dr3f2_glim["DEC"]+dec_med_diff_f2,deep2f2["RA_DEEP"], deep2f2["DEC_DEEP"], tol=1./3600.)
# Save unmatched DEEP2
f2_only_deep2 = deep2f2[np.setdiff1d(np.arange(deep2f2["RA_DEEP"].size,dtype=int),idx2)]
save_fits(f2_only_deep2, "unmatched-deep2-f2-photo-redz-oii-glim24.fits")
# Append DEEP2 info for matched DR3 objects.
append_list = [('OBJNO', '>i4'), ('BESTB', '>f4'), ('BESTR', '>f4'), ('BESTI', '>f4'), ('BESTBERR', '>f4'), ('BESTRERR', '>f4'), ('BESTIERR', '>f4'), ('BADFLAG', 'u1'), ('OII_3727', '>f8'), ('OII_3727_ERR', '>f8'), ('RED_Z', '>f8'), ('Z_ERR', '>f8'), ('ZQUALITY', '>f8'), ('TARG_WEIGHT', '>f8'), ('weight', '>f8'), ('BRI_cut', '>f8'), ('cn', '>f8')]
dr3f2_glim_deep2 = None
for e in append_list:
    if dr3f2_glim_deep2 is None:
        dr3f2_glim_deep2 = fits_append(dr3f2_glim, deep2f2[e[0]], e[0], idx1, idx2)
    else:
        dr3f2_glim_deep2 = fits_append(dr3f2_glim_deep2, deep2f2[e[0]], e[0], idx1, idx2)
# One more field to indicate whether DR3 object was found in DEEP2 or not. If found 1.
DEEP2_matched = np.zeros(dr3f2_glim_deep2.shape[0], dtype=int)
DEEP2_matched[idx1] = 1
dr3f2_glim_deep2 = rec.append_fields(dr3f2_glim_deep2, "DEEP2_matched", DEEP2_matched, dtypes=DEEP2_matched.dtype, usemask=False, asrecarray=True)
save_fits(dr3f2_glim_deep2,"DECaLS-DR3-DEEP2f2-glim24.fits")
print("Completed.\n")


print("Field 3")
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f3_glim["RA"]+ra_med_diff_f3,dr3f3_glim["DEC"]+dec_med_diff_f3,deep2f3["RA_DEEP"], deep2f3["DEC_DEEP"], tol=1./3600.)
# check_astrometry(dr3f3_glim["RA"][idx1]+ra_med_diff_f3,dr3f3_glim["DEC"][idx1]+dec_med_diff_f3,deep2f3["RA"][idx2], deep2f3["DEC"][idx2],pt_size=0.1)
# Save unmatched DEEP2
f3_only_deep2 = deep2f3[np.setdiff1d(np.arange(deep2f3["RA_DEEP"].size,dtype=int),idx2)]
save_fits(f3_only_deep2, "unmatched-deep2-f3-photo-redz-oii-glim24.fits")
# Append DEEP2 info for matched DR3 objects.
append_list = [('OBJNO', '>i4'), ('BESTB', '>f4'), ('BESTR', '>f4'), ('BESTI', '>f4'), ('BESTBERR', '>f4'), ('BESTRERR', '>f4'), ('BESTIERR', '>f4'), ('BADFLAG', 'u1'), ('OII_3727', '>f8'), ('OII_3727_ERR', '>f8'), ('RED_Z', '>f8'), ('Z_ERR', '>f8'), ('ZQUALITY', '>f8'), ('TARG_WEIGHT', '>f8'), ('weight', '>f8'), ('BRI_cut', '>f8'), ('cn', '>f8')]
dr3f3_glim_deep2 = None
for e in append_list:
    if dr3f3_glim_deep2 is None:
        dr3f3_glim_deep2 = fits_append(dr3f3_glim, deep2f3[e[0]], e[0], idx1, idx2)
    else:
        dr3f3_glim_deep2 = fits_append(dr3f3_glim_deep2, deep2f3[e[0]], e[0], idx1, idx2)
# One more field to indicate whether DR3 object was found in DEEP2 or not. If found 1.
DEEP2_matched = np.zeros(dr3f3_glim_deep2.shape[0], dtype=int)
DEEP2_matched[idx1] = 1
dr3f3_glim_deep2 = rec.append_fields(dr3f3_glim_deep2, "DEEP2_matched", DEEP2_matched, dtypes=DEEP2_matched.dtype, usemask=False, asrecarray=True)
save_fits(dr3f3_glim_deep2,"DECaLS-DR3-DEEP2f3-glim24.fits")
print("Completed.\n")


print("Field 4")
idx1, idx2 = crossmatch_cat1_to_cat2(dr3f4_glim["RA"]+ra_med_diff_f4,dr3f4_glim["DEC"]+dec_med_diff_f4,deep2f4["RA"], deep2f4["DEC"], tol=1./3600.)
# check_astrometry(dr3f4_glim["RA"][idx1]+ra_med_diff_f4,dr3f4_glim["DEC"][idx1]+dec_med_diff_f4,deep2f4["RA"][idx2], deep2f4["DEC"][idx2],pt_size=0.1)
# Save unmatched DEEP2
f4_only_deep2 = deep2f4[np.setdiff1d(np.arange(deep2f4["RA"].size,dtype=int),idx2)]
save_fits(f4_only_deep2, "unmatched-deep2-f4-photo-redz-oii-glim24.fits")
# Append DEEP2 info for matched DR3 objects.
append_list = [('OBJNO', '>i4'), ('MAGB', '>f4'), ('MAGR', '>f4'), ('MAGI', '>f4'), ('MAGBERR', '>f4'), ('MAGRERR', '>f4'), ('MAGIERR', '>f4'), ('BADFLAG', 'u1'), ('OII_3727', '>f8'), ('OII_3727_ERR', '>f8'), ('RED_Z', '>f8'), ('Z_ERR', '>f8'), ('ZQUALITY', '>f8'), ('TARG_WEIGHT', '>f8'), ('weight', '>f8'), ('BRI_cut', '>f8'), ('cn', '>f8')]
dr3f4_glim_deep2 = None
for e in append_list:
    if dr3f4_glim_deep2 is None:
        dr3f4_glim_deep2 = fits_append(dr3f4_glim, deep2f4[e[0]], e[0], idx1, idx2)
    else:
        dr3f4_glim_deep2 = fits_append(dr3f4_glim_deep2, deep2f4[e[0]], e[0], idx1, idx2)
# One more field to indicate whether DR3 object was found in DEEP2 or not. If found 1.
DEEP2_matched = np.zeros(dr3f4_glim_deep2.shape[0], dtype=int)
DEEP2_matched[idx1] = 1
dr3f4_glim_deep2 = rec.append_fields(dr3f4_glim_deep2, "DEEP2_matched", DEEP2_matched, dtypes=DEEP2_matched.dtype, usemask=False, asrecarray=True)
save_fits(dr3f4_glim_deep2,"DECaLS-DR3-DEEP2f4-glim24.fits")
print("Completed.\n")




##############################################################################
print("4. Estimating DR3 area after imposing DEEP2 mask and Tycho-2 stellar mask \n\
    (based on the fraction of objects masked.)")
# Field 2
area_f2 = est_spec_area("windowf.21.fits")+est_spec_area("windowf.22.fits")
# print(area_f2)
# Field 3
area_f3 = est_spec_area("windowf.31.fits")+est_spec_area("windowf.32.fits")+est_spec_area("windowf.33.fits")
# Field 4
area_f4 = est_spec_area("windowf.41.fits")+est_spec_area("windowf.42.fits")


# Making correction according to # of objects masked by star mask.
print("Field 2")
fname ="DECaLS-DR3-Tractor-DEEP2f2.fits"
table = load_fits_table(fname)
area_f2 *= true_false_fraction(load_star_mask(table))[-1]
print("Intersection area: %.4f" % area_f2)

print("Field 3")
fname ="DECaLS-DR3-Tractor-DEEP2f3.fits"
table = load_fits_table(fname)
area_f3 *= true_false_fraction(load_star_mask(table))[-1]
print("Intersection area: %.4f" % area_f3)

print("Field 4")
fname ="DECaLS-DR3-Tractor-DEEP2f4.fits"
table = load_fits_table(fname)
area_f4 *= true_false_fraction(load_star_mask(table))[-1]
print("Intersection area: %.4f" % area_f4)

area = [area_f2,area_f3,area_f4]
np.savetxt("intersection-area-f234",area)
print("Completed.\n")
