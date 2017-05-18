# Loading modules
import numpy as np
from os import listdir
from os.path import isfile, join
from astropy.io import ascii, fits
from astropy.wcs import WCS
import numpy.lib.recfunctions as rec
from xd_elg_utils import *
import sys

large_random_constant = -999119283571
deg2arcsec=3600

data_directory = "./"


# True if tractor files have been already downloaded.
tractor_file_downloaded = True


##############################################################################
if not tractor_file_downloaded: # If the tractor files are not downloaded.
	print("1. Generate download scripts for relevant Tractor files.")
	print("This step generates three files that the user can use to download\n\
	the relevant tractor files.")
	print("To identify relevant bricks use survey-bricks-dr3.fits which the user\n\
	should have downloaded. Approximate field ranges.\n\
	\n\
	Field 2\n\
	RA bounds: [251.3, 253.7]\n\
	DEC bounds: [34.6, 35.3]\n\
	\n\
	Field 3\n\
	RA bounds: [351.25, 353.8]\n\
	DEC bounds: [-.2, .5]\n\
	\n\
	Field 4\n\
	RA bounds: [36.4, 38]\n\
	DEC bounds: [.3, 1.0]\n\
	")

	fits_bricks = fits.open(data_directory+"survey-bricks-dr3.fits")[1].data
	ra = fits_bricks['ra'][:]
	dec = fits_bricks['dec'][:]
	br_name = fits_bricks['brickname'][:]

	# Getting the brick names near the ranges specified below.
	tol = 0.25
	f2_bricks = return_bricknames(ra, dec, br_name,[251.3, 253.7],[34.6, 35.3],tol)
	f3_bricks = return_bricknames(ra, dec, br_name,[351.25, 353.8],[-.2, .5],tol)
	f4_bricks = return_bricknames(ra, dec, br_name,[36.4,38.],[.3, 1.0],tol)
	bricks = [f2_bricks, f3_bricks, f4_bricks]

	print("Generating download scripts. DR3-DEEP2f**-tractor-download.sh")
	portal_address = "http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr3/tractor/"
	postfix = ".fits\n"
	prefix = "wget "
	for i in range(3):
	    f = open("DR3-DEEP2f%d-tractor-download.sh"%(i+2),"w")
	    for brick in bricks[i]:
	        tractor_directory = brick[:3]
	        brick_address = tractor_directory+"/tractor-"+brick+postfix
	        download_command = prefix + portal_address + brick_address
	        f.write(download_command)
	    f.close()
	print("Completed")
	print("Exiting the program. Please download the necessary files using the script\n\
		and re-run the program with tractor_file_downloaded=True.")
	sys.exit()
else:
	print("Proceeding using the downloaded tractor files.")
	print("Within data_directory, Tractor files should be \n\
		saved in directories in \DR3-f**\.")


##############################################################################	
print("2. Combine all Tractor files by field, append Tycho-2 stellar mask column, \n\
	and mask objects using DEEP2 window funtions.")
print("2a. Combining the tractor files: Impose mask conditions (brick_primary==True\n\
	and flux inverse variance positive).")
# Field 2
DR3f2 = combine_tractor(data_directory+"DR3-f2/")
# Field 3
DR3f3 = combine_tractor(data_directory+"DR3-f3/")
# Field 4
DR3f4 = combine_tractor(data_directory+"DR3-f4/")
print("Completed.")


print("2b. Append Tycho2 stark mask field.")
# Field 2 
DR3f2 = apply_tycho(DR3f2,"tycho2.fits",galtype="ELG")
# Field 3
DR3f3 = apply_tycho(DR3f3,"tycho2.fits",galtype="ELG")
# Field 4 
DR3f4 = apply_tycho(DR3f4,"tycho2.fits",galtype="ELG")
print("Completed.")

print("2c. Impose DEEP2 window functions.")
# Field 2
idx = np.logical_or(window_mask(DR3f2["ra"], DR3f2["dec"], "windowf.21.fits"), window_mask(DR3f2["ra"], DR3f2["dec"], "windowf.22.fits"))
DR3f2_trimmed = DR3f2[idx]

# Field 3
idx = np.logical_or.reduce((window_mask(DR3f3["ra"], DR3f3["dec"], "windowf.31.fits"), window_mask(DR3f3["ra"], DR3f3["dec"], "windowf.32.fits"),window_mask(DR3f3["ra"], DR3f3["dec"], "windowf.33.fits")))
DR3f3_trimmed = DR3f3[idx]

# Field 4
idx = np.logical_or(window_mask(DR3f4["ra"], DR3f4["dec"], "windowf.41.fits"), window_mask(DR3f4["ra"], DR3f4["dec"], "windowf.42.fits"))
DR3f4_trimmed = np.copy(DR3f4[idx])
print("Completed.")


##############################################################################	
print("3. Save the trimmed catalogs.")
# Field 2
cols = fits.ColDefs(DR3f2_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('DECaLS-DR3-Tractor-DEEP2f2.fits', clobber=True)

# Field 3
cols = fits.ColDefs(DR3f3_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('DECaLS-DR3-Tractor-DEEP2f3.fits', clobber=True)

# Field 4
cols = fits.ColDefs(DR3f4_trimmed)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('DECaLS-DR3-Tractor-DEEP2f4.fits', clobber=True)
print("Completed.")

