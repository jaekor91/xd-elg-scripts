import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
import numpy.lib.recfunctions as rec
from os import listdir
from os.path import isfile, join

large_random_constant = -999119283571
deg2arcsec=3600


def window_mask(ra, dec, w_fname):
    """
    Given the ra,dec of objects and a window function file name, the function 
    returns an boolean array whose elements equal True when the corresponding 
    objects lie within regions where the map is positive.
    
    Note: the windowf.**.fits files were not set up in a convenient format 
    so below I had to perform a simple but tricky affine transformation. 
    """
    
    # Import the window map and get pixel limits.
    window = fits.open(w_fname)[0].data.T # Note that transpose.
    px_lim = window.shape[1]
    py_lim = window.shape[0]      
    
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Convert ra/dec to pixel values and round.
    px, py = w.wcs_world2pix(ra, dec, 0)
    px_round = np.round(py)
    py_round = np.round(px)
  
    # Creating the array.
    idx = np.zeros(px_round.size, dtype=bool)
    for i in range(px.size):
        if (px_round[i]>=0) and (px_round[i]<px_lim) and (py_round[i]>=0) and (py_round[i]<py_lim): # Check if the object lies within the window frame. 
            if (window[py_round[i],px_round[i]]>0): # Check if the object is in a region where there is spectroscopy.
                idx[i] = True
    
    return idx


def est_spec_area(w_fname):
    """
    The following function estiamtes the spectroscopic area given a window 
    function file.
    """
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Importing the window
    window = fits.open(w_fname)[0].data
    px_lim = window.shape[0]
    py_lim = window.shape[1]
   
    
    # Convert ra/dec to pixel values and round.
    ra, dec = w.wcs_pix2world([0,px_lim], [0,py_lim], 0)
    
    # Calculating the area
    area = (ra[1]-ra[0])*(dec[1]-dec[0])
    
    # Calculating the fraction covered by spectroscopy
    frac = (window>0).sum()/(px_lim*py_lim+1e-12)
    
    return frac*area
        
        
    
def import_zcat(z_fname):
    """
    Given DEEP2 redshift catalog filename, import and return relevant fields.
    """
    data = fits.open(z_fname)[1].data
    
    return data["OBJNO"], data["RA"], data["DEC"], data["OII_3727"], data["OII_3727_ERR"], data["ZHELIO"], data["ZHELIO_ERR"], data["ZQUALITY"], data["TARG_WEIGHT"]
    


def match_objno(objno1, objno2):
    """
    Given two objno arrays, return idx of items that match. This algorithm can be slow, O(N^2), but it should work.
    The input arrays have to be a set, meaning a list of unique items.
    """    
    global large_random_constant
    # Finding the intersection
    intersection = np.intersect1d(objno1, objno2)
    print("# of elements in intersection: %d"% intersection.size)
    
    # Creating placeholders for idx's to be returned.
    idx1 = np.ones(intersection.size,dtype=int)*large_random_constant
    idx2 = np.ones(intersection.size,dtype=int)*large_random_constant
    
    # Creating objno1, objno2 copies with integer tags before sorting.
    
    objno1_tagged = np.rec.fromarrays((objno1, range(objno1.size)),dtype=[('id', int), ('tag', int)])
    objno2_tagged = np.rec.fromarrays((objno2, range(objno2.size)),dtype=[('id', int), ('tag', int)])
    
    # Sorting according id
    objno1_tagged = np.sort(objno1_tagged, axis=0, order="id")
    objno2_tagged = np.sort(objno2_tagged, axis=0, order="id")
    
    # tags
    tags1 = objno1_tagged["tag"]
    tags2 = objno2_tagged["tag"]
    
    # values
    objno1_vals = objno1_tagged["id"]    
    objno2_vals = objno2_tagged["id"]
        
    # For each id in the intersection set, find the corresponding indices in objno1 and objno2 and save. 
    for i,e in enumerate(intersection):
        idx1[i] = tags1[np.searchsorted(objno1_vals,e)]
        idx2[i] = tags2[np.searchsorted(objno2_vals,e)]
    
    return idx1, idx2


def pcat_append(pcat, new_col, col_name, idx1, idx2):
    """
    Given DEEP2 pcat recarray and a pair of a field column, and a field name,
    append the new field column to the recarray using OBJNO-matched values and name the appended column the field name.
    Must provide appropriate idx values for both pcats and additional catalogs.
    """
    global large_random_constant
    new_col_sorted = np.ones(pcat.shape[0])*large_random_constant
    new_col_sorted[idx1] = new_col[idx2]
    
    new_pcat = rec.append_fields(pcat, col_name, new_col_sorted, dtypes=new_col_sorted.dtype, usemask=False, asrecarray=True)
    return new_pcat
    


def generate_class_col(pcat):
    """
    Given a pcat array with required fields, produce a column that classify objects into different classes.
    """
    # Extracting columns
    OII = pcat["OII_3727"]*1e17
    Z = pcat["RED_Z"]
    ZQUALITY = pcat["ZQUALITY"]
    OII_ERR = pcat["OII_3727_ERR"]
    BRIcut = pcat["BRI_cut"]
    
    
    # Placeholder for the class column.
    class_col = np.ones(pcat.shape[0],dtype=int)*large_random_constant
    
    # Gold, CN=0: OII>8, Z in [1.1, 1.6]
    ibool = (OII>8) & (Z>1.1) & (Z<1.6)  & (BRIcut==1) & (ZQUALITY>=3) & (OII_ERR>0) 
    class_col[ibool] = 0
    
    # Silver, CN=1: OII>8, Z in [0.6, 1.1]
    ibool = (OII>8) & (Z>0.6) & (Z<1.1) & (OII_ERR>0)  & (BRIcut==1) & (ZQUALITY>=3)
    class_col[ibool] = 1

    # LowOII, CN=2: OII<8, Z in [0.6, 1.6]
    ibool =  (OII<8) & (Z>0.6) & (Z<1.6)  & (OII_ERR>0) & (ZQUALITY>=3) & (BRIcut==1)
    class_col[ibool] = 2
    # Note that many ELG objects were assigned negative OII, which are unphysical. 

    # NoOII, CN=3: OII=?, Z in [0.6, 1.6] and secure redshift
    ibool = (Z>0.6) & (Z<1.6) & (OII_ERR<=0) & (ZQUALITY>=3) & (BRIcut==1)    
    class_col[ibool] = 3

    # LowZ, CN=4: OII=NA, Z outside [0.6, 1.6]
    ibool = np.logical_or((np.logical_or((Z>1.6), (Z<0.6)) & (ZQUALITY>=3)),(ZQUALITY==-1))  & (OII_ERR<=0) & (BRIcut==1)
    class_col[ibool] = 4

    # NoZ, CN=5: OII=NA, Z undetermined.
    ibool = np.logical_or.reduce(((ZQUALITY==-2) , (ZQUALITY==0) , (ZQUALITY==1) ,(ZQUALITY==2)))& (BRIcut==1)  & (OII_ERR<=0)
    class_col[ibool] = 5
    
    # D2reject, CN=6
    ibool = BRIcut!=1
    class_col[ibool] = 6
    
    # D2unobserved, CN=8
    ibool = (BRIcut==1) & (ZQUALITY<-10)   # Objects that were not assigned color-selection flag are classifed as DEEP2 color rejected objects.
    class_col[ibool] = 8  
    
    return class_col

def count_nn(arr):
    """
    Count the number of non-negative elements.
    """   
    return arr[arr>-1].size




def class_breakdown(fn, cn, weight, area, rwd="D"):
    """
    Given a list of class fields and corresponding weights and areas, return the breakdown of object 
    for each class. fn gives the field number. 
    """
    
    # Place holder for tallying
    counts = np.zeros(8)
    
    # Generate counts
    for i in range(len(fn)):
        # Computing counts
        if rwd == "R":
            tmp = generate_raw_breakdown(cn[i])
        elif rwd == "W":
            tmp = generate_weighted_breakdown(cn[i], weight[i])
        else:
            tmp = generate_density_breakdown(cn[i], weight[i], area[i])
        
        # Tallying counts
        if rwd in ["R", "W"]:
            counts += tmp
        else:
            counts += tmp/len(fn)
        
        # Printing counts
        print(str_counts(fn[i], rwd, tmp))            

    # Total or average counts
    if rwd in ["R", "W"]:
        print(str_counts("Total", rwd, counts))
    else:
        print(str_counts("Avg.", rwd, counts))


    
def str_counts(fn, rwd_str, counts):
    """
    Given the counts of various class of objects return a formated string.
    """
    if type(fn)==str:
        return_str = "%s & %s " % (fn,rwd_str)
    else:
        return_str = "%d & %s " % (fn,rwd_str)
        
    for i in range(counts.size):
        return_str += "& %d " % counts[i]
    
    return_str += "& %d " % np.sum(counts)
    
    return_str += latex_eol()
    
    return return_str
    
    
def generate_raw_breakdown(cn):
    return np.delete(np.bincount(cn.astype(int)),7)

def generate_weighted_breakdown(cn, weight):
    counts = np.zeros(8,dtype=int)
    for i,e in enumerate(np.delete(np.arange(0,9,1, dtype=int),7)):
        if (e!=6) and (e!=8):
            counts[i] = np.sum(weight[cn==e])
        else:
            counts[i] = np.sum(cn==e)
    return counts

def generate_density_breakdown(cn, weight,area):
    counts = np.zeros(8,dtype=int)
    for i,e in enumerate(np.delete(np.arange(0,9,1, dtype=int),7)):
        if (e!=6) and (e!=8):
            counts[i] = np.sum(weight[cn==e])/np.float(area)
        else:
            counts[i] = np.sum(cn==e)/np.float(area)
    return counts
    

def generate_table_header():
    return "Field & R/W/D & "+" & ".join(class_names()) + " & Total" + latex_eol() + latex_hline()

def latex_eol():
    return "\\\\ \\hline"

def latex_hline():
    return "\\hline"
    
def class_names():
    """
    Provide a list of class names.
    """
    return ["Gold", "Silver", "LowOII","NoOII", "LowZ", "NoZ", "D2reject", "D2unobserved"]



def return_bricknames(ra, dec, br_name, ra_range, dec_range,tol):
    ibool = (ra>(ra_range[0]-tol)) & (ra<(ra_range[1]+tol)) & (dec>(dec_range[0]-tol)) & (dec<(dec_range[1]+tol))
    return  br_name[ibool]






def combine_tractor(fits_directory):
    """
    Given the file directory, find all Tractor fits files combine them and return as a rec-array.
    """
    onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
    print("Number of files in %s %d" % (fits_directory, len(onlyfiles)-1))
    
    
    DR3 = None
    for i,e in enumerate(onlyfiles,start=1):
        # If the file ends with "fits"
        if e[-4:] == "fits":
            print("Combining file %d. %s" % (i,e))
            # If DR3 has been set with something.
            tmp_table = apply_mask(fits.open(fits_directory+e)[1].data)
            if DR3 is not None:
                DR3 = np.hstack((DR3, tmp_table))
            else:
                DR3 = tmp_table
                
    return DR3

def apply_mask(table):
    """
    Given a tractor catalog table, apply the standard mask. brick_primary and flux inverse variance. 
    """
    brick_primary = load_brick_primary(table)
    givar, rivar, zivar = load_grz_invar(table)
    ibool = (brick_primary==True) & (givar>0) & (rivar>0) &(zivar>0) 
    table_trimmed = np.copy(table[ibool])

    return table_trimmed
    
def load_grz_anymask(fits):
    g_anymask = fits['DECAM_ANYMASK'][:][:,1]
    r_anymask = fits['DECAM_ANYMASK'][:][:,2]
    z_anymask = fits['DECAM_ANYMASK'][:][:,4]
    
    return g_anymask, r_anymask, z_anymask

def load_grz_allmask(fits):
    g_allmask = fits['DECAM_ALLMASK'][:][:,1]
    r_allmask = fits['DECAM_ALLMASK'][:][:,2]
    z_allmask = fits['DECAM_ALLMASK'][:][:,4]
    
    return g_allmask, r_allmask, z_allmask


def load_radec(fits):
    ra = fits["ra"][:]
    dec= fits["dec"][:]
    return ra, dec


def load_brick_primary(fits):
    return fits['brick_primary'][:]


def load_shape(fits):
    r_dev = fits['SHAPEDEV_R'][:]
    r_exp = fits['SHAPEEXP_R'][:]
    return r_dev, r_exp

def load_grz_invar(fits):
    givar = fits['decam_flux_ivar'][:][:,1]
    rivar = fits['DECAM_FLUX_IVAR'][:][:,2]
    zivar = fits['DECAM_FLUX_IVAR'][:][:,4]
    return givar, rivar, zivar

def load_star_mask(table):
    return table["TYCHOVETO"][:].astype(int).astype(bool)


def load_grz(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    g = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,1]/fits['decam_mw_transmission'][:][:,1]))
    r = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,2]/fits['decam_mw_transmission'][:][:,2]))
    z = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,4]/fits['decam_mw_transmission'][:][:,4]))
    return g, r, z    

def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data

def save_fits(data, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    cols = fits.ColDefs(np.copy(data)) # This is somehow necessary.
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 

def save_fits_join(data1,data2, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    
    data = rec.merge_arrays((data1,data2), flatten=True, usemask=False,asrecarray=True)
    cols = fits.ColDefs(data) 
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 
    



##############################################################################
# The following is adpated from the URL indicated below.
# """
#     ImagingLSS
#     https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

#     veto objects based on a star catalogue.
#     The tycho vetos are based on the email discussion at:
#     Date: June 18, 2015 at 3:44:09 PM PDT
#     To: decam-data@desi.lbl.gov
#     Subject: decam-data Digest, Vol 12, Issue 29
#     These objects takes a decals object and calculates the
#     center and rejection radius for the catalogue in degrees.
#     Note : The convention for veto flags is True for 'reject',
#     False for 'preserve'.

#     apply_tycho takes the galaxy catalog and appends a Tychoveto column
#     the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
# """

# Import modules
import sys

def BOSS_DR9(tycho):
    bmag = tycho['BMAG']
    # BOSS DR9-11
    b = bmag.clip(6, 11.5)
    R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
    return R

def DECAM_LRG(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
    return R

DECAM_ELG = DECAM_LRG

def DECAM_QSO(tycho):
    vtmag = tycho['VTMAG']
    # David Schlegel recommends not applying a bright star mask
    return vtmag - vtmag

def DECAM_BGS(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
    return R

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = np.empty(len(ra), dtype=('f8', 3))
    ra = ra * (np.pi / 180)
    dec = dec * (np.pi / 180)
    pos[:, 2] = np.sin(dec)
    pos[:, 0] = np.cos(dec) * np.sin(ra)
    pos[:, 1] = np.cos(dec) * np.cos(ra)
    return pos

def tycho(filename):
    """
    read the Tycho-2 catalog and prepare it for the mag-radius relation
    """
    dataf = fits.open(filename)
    data = dataf[1].data
    tycho = np.empty(len(data),
        dtype=[
            ('RA', 'f8'),
            ('DEC', 'f8'),
            ('VTMAG', 'f8'),
            ('VMAG', 'f8'),
            ('BMAG', 'f8'),
            ('BTMAG', 'f8'),
            ('VARFLAG', 'i8'),
            ])
    tycho['RA'] = data['RA']
    tycho['DEC'] = data['DEC']
    tycho['VTMAG'] = data['MAG_VT']
    tycho['BTMAG'] = data['MAG_BT']
    vt = tycho['VTMAG']
    bt = tycho['BTMAG']
    b = vt - 0.09 * (bt - vt)
    v = b - 0.85 * (bt - vt)
    tycho['VMAG']=v
    tycho['BMAG']=b
    dataf.close()
    return tycho


def txts_read(filename):
    obj = np.loadtxt(filename)
    typeobj = np.dtype([
              ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
              ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
              ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
              ('W2flux','f4'), ('W2noise','f4')
              ])
    nobj = obj[:,0].size
    data = np.zeros(nobj, dtype=typeobj)
    data['RA'][:] = obj[:,0]
    data['DEC'][:] = obj[:,1]
    data['COMPETENESS'][:] = obj[:,2]
    data['rflux'][:] = obj[:,3]
    data['rnoise'][:] = obj[:,4]
    data['gflux'][:] = obj[:,5]
    data['gnoise'][:] = obj[:,6]
    data['zflux'][:] = obj[:,7]
    data['znoise'][:] = obj[:,8]
    data['W1flux'][:] = obj[:,9]
    data['W1noise'][:] = obj[:,10]
    data['W2flux'][:] = obj[:,11]
    data['W2noise'][:] = obj[:,12]
    #datas = np.sort(data, order=['RA'])
    return data

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.
        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees
        Returns
        -------
        Vetomask : True for veto, False for keep.
    """
    from sklearn.neighbors import KDTree
    pos_stars = radec2pos(center[0], center[1])
    R = 2 * np.sin(np.radians(R) * 0.5)
    pos_obj = radec2pos(coord[0], coord[1])
    tree = KDTree(pos_obj)
    vetoflag = ~np.zeros(len(pos_obj), dtype='?')
    arg = tree.query_radius(pos_stars, r=R)
    arg = np.concatenate(arg)
    vetoflag[arg] = False
    return vetoflag



def apply_tycho(objgal, tychofn,galtype='LRG'):
    # reading tycho star catalogs
    tychostar = tycho(tychofn)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['RA'], tychostar['DEC'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['ra'], objgal['dec'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rec.append_fields(objgal, ['TYCHOVETO'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal
