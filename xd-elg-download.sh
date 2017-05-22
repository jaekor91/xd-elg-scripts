# All files referred to here as well as under the Data repostiory section in README 
# must be downloaded before the training scripts are run.

# Download DEEP2 photometry files
# Note 1: For Field 4, a combined and cleaned catalog was obtained from Jeff Newman.
# 		  The file is available on the data repository. Filename: 
# Note 2: Files were found here: http://deep.ps.uci.edu/DR4/photo.extended.html
wget http://deep.ps.uci.edu/DR4/data/pcat_ext.21.fits.gz
wget http://deep.ps.uci.edu/DR4/data/pcat_ext.22.fits.gz
wget http://deep.ps.uci.edu/DR4/data/pcat_ext.31.fits.gz
wget http://deep.ps.uci.edu/DR4/data/pcat_ext.32.fits.gz
wget http://deep.ps.uci.edu/DR4/data/pcat_ext.33.fits.gz
# wget http://deep.ps.uci.edu/DR4/data/pcat_ext.41.fits.gz
# wget http://deep.ps.uci.edu/DR4/data/pcat_ext.42.fits.gz
# wget http://deep.ps.uci.edu/DR4/data/pcat_ext.43.fits.gz


# Download DEEP2 window functions
wget http://deep.ps.uci.edu/DR4/data/windowf.21.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.22.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.31.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.32.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.33.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.41.fits.gz
wget http://deep.ps.uci.edu/DR4/data/windowf.42.fits.gz


# Download survey-bricks-dr3.fits
wget http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr3/survey-bricks-dr3.fits.gz


# For downloading Tractor files, use
source DR3-DEEP2f2-tractor-download.sh	
source DR3-DEEP2f3-tractor-download.sh	
source DR3-DEEP2f4-tractor-download.sh