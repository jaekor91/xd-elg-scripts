import numpy as np
import matplotlib.pyplot as plt
# For numba
import os
os.environ['NUMBA_NUM_THREADS'] = "1"
import numba as nb

# matplotlib ticks
import matplotlib as mpl 
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1.5


cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]

def apply_XD_globalerror(objs, last_FoM, param_directory, glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5,reg_r=5e-4/(0.025**2 * 0.05),f_i=[1., 1., 0., 0.25, 0., 0.25, 0.],\
                       gmin = 21., gmax = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1]):
    """ Apply ELG XD selection. Default uses fiducial set of parameters.

    Args:
        objs: A DECaLS fits table or the following list of extracted columns. 
            objs = [g,r,z, givar, rivar, zivar, gflux, rflux, zflux]
        last_FoM: Threshold FoM.
        param_directory: Directory where parameter files named in expected formats are to be found.
        
    Optional:
        glim, rlim, zlim: 5-sigma detection limiting magnitudes. 
        gr_ref, rz_ref: Number density conserving global error reference point.
        reg_r: Regularization parameter. Empirically set to avoid pathologic 
            behaviors of the selection boundary.
        f_i: Various class weights for FoM.
        gmin, gmax: Minimum and maximum g-magnitude range to consider.
        K_i and dNdm_type: See generate_XD_model_dictionary() doc string.
    
    Returns:
        iXD: Boolean mask array that implements XD selection.
        FoM: Figure of Merit number computed for objects that pass the initial set of masks.

    Note:
        1. The current version of XD selection method assumes the imposition of decam_allmask 
            and tycho2 stellar mask. (The individual class densities have been fitted with these 
            masks imposed.) However, the code does not implement them yet as we want to understand
            the large scale systematics of the XD selection with and without these masks.
        2. A different version of this function using individual Tractor error is called 
            apply_XD_Tractor_error().
            
    Process in summary:
        - Construct a Python dictionary that contains all XD GMM and dNdm parameters
            using a string.
        - Load variables from the input fits table.
        - Compute which objects pass the reasonable imaging quality cut 
            (SNR>2, flux positive, and flux invariance positive).
        - Compute which objects pass a rough color cut that eliminates a
            bulk of low redshift contaiminants. 
        - For each object that passes the above two cuts, compute Figure of Merit FoM.
        - If FoM>FoM_last, then include the object in the selection.
        - Append this selection column to the table and return.
    """

    params = generate_XD_model_dictionary(param_directory, K_i=K_i, dNdm_type=dNdm_type)

    ####### Load variables. #######
    if type(objs) is list:
        g,r,z, givar, rivar, zivar, gflux, rflux, zflux = objs        
    else:
        # Flux
        gflux = objs['DECAM_FLUX'][:][:,1]/objs['DECAM_MW_TRANSMISSION'][:][:,1] 
        rflux = objs['DECAM_FLUX'][:][:,2]/objs['DECAM_MW_TRANSMISSION'][:][:,2]
        zflux = objs['DECAM_FLUX'][:][:,4]/objs['DECAM_MW_TRANSMISSION'][:][:,4]
        # mags
        #ADM added explicit capture of runtime warnings for zero and negative fluxes
        with np.errstate(invalid='ignore',divide='ignore'):
            g = (22.5 - 2.5*np.log10(gflux)) 
            r = (22.5 - 2.5*np.log10(rflux))
            z = (22.5 - 2.5*np.log10(zflux))
            # Inver variance
            givar = objs['DECAM_FLUX_IVAR'][:][:,1]
            rivar = objs['DECAM_FLUX_IVAR'][:][:,2]
            zivar = objs['DECAM_FLUX_IVAR'][:][:,4]

    # Color
    rz = (r-z); gr = (g-r)    

    ####### Reaonsable quaity cut. #######
    iflux_positive = (gflux>0)&(rflux>0)&(zflux>0)
    ireasonable_color = (gr>-0.5) & (gr<2.5) & (rz>-0.5) &(rz<2.7) & (g<gmax) & (g>gmin)
    thres = 2
    igrz_SN2 =  ((gflux*np.sqrt(givar))>thres)&((rflux*np.sqrt(rivar))>thres)&((zflux*np.sqrt(zivar))>thres)
    # Combination of above cuts.
    ireasonable = iflux_positive & ireasonable_color & igrz_SN2

    ####### A rough cut #######
    irough = (gr<1.3) & np.logical_or(gr<(rz+0.3) ,gr<0.3)

    ####### Objects for which FoM to be calculated. #######
    ibool = ireasonable & irough 
        
    
    ######## Compute FoM values for objects that pass the cuts. #######
    # Place holder for FoM
    FoM = np.zeros(ibool.size, dtype=np.float)

    # Select subset of objects.
    mag = g[ibool]
    flux = gflux[ibool]    
    gr = gr[ibool]
    rz = rz[ibool]

    # Compute the global error noise corresponding to each objects.
    const = 2.5/(5*np.log(10)) 
    gvar = (const * 10**(0.4*(mag-glim)))**2
    rvar = (const * 10**(0.4*(mag-gr_ref-rlim)))**2
    zvar = (const * 10**(0.4*(mag-gr_ref-rz_ref-zlim)))**2        
 
     # Calculate the densities.
    FoM_num = np.zeros_like(gr)
    FoM_denom = np.zeros_like(gr)
    for i in range(7): # number of classes.
        n_i = GMM_vectorized(rz, gr, params[i, "amp"], params[i, "mean"],params[i, "covar"], gvar, rvar, zvar)  * dNdm(params[(i,"dNdm")], flux)
        FoM_num += f_i[i]*n_i
        FoM_denom += n_i
           
    FoM[ibool] = FoM_num/(FoM_denom+reg_r+1e-12) # For proper broadcasting.
    
    # XD-selection
    iXD = FoM>last_FoM
    
    return iXD, FoM

# Helper function 1.
def GMM_vectorized(gr, rz, amps, means, covars, gvar, rvar, zvar):
    """
    Color-color density.

    Note 1: This routine was originally written based on gr vs. rz convention. However, I decided to adopt
    rz vs. gr convention and therefore, the user must provide "gr=rz, rz=gr" as input in order for this
    function to work properly.

    Params
    ------
    gvar, rvar, zvar: Pre-computed errors based on individual grz values scaled from 5-sigma detection limits.
    """
    # Place holder for return array.
    density = np.zeros(gr.size,dtype=np.float)
    
    # Compute 
    for i in range(amps.size):
        # Calculating Sigma+Error
        C11 = covars[i][0,0]+gvar+rvar
        C12 = covars[i][0,1]+rvar
        C22 = covars[i][1,1]+rvar+zvar
        
        # Compute the determinant
        detC = C11*C22-C12**2
        
        # Compute variables
        x11 = (gr-means[i][0])**2
        x12 = (gr-means[i][0])*(rz-means[i][1])
        x22 = (rz-means[i][1])**2
        
        # Calculating the exponetial
        EXP = np.exp(-(C22*x11-2*C12*x12+C11*x22)/(2.*detC+1e-12))
        
        density += amps[i]*EXP/(2*np.pi*np.sqrt(detC)+1e-12)
    
    return density


# Helper function 2.
def dNdm(params, flux):
    num_params = params.shape[0]
    if num_params == 2:
        return pow_law(params, flux)
    elif num_params == 4:
        return broken_pow_law(params, flux)

# Helper function 3.
def pow_law(params, flux):
    A = params[1]
    alpha = params[0]
    return A* flux**alpha

# Helper function 4.
def broken_pow_law(params, flux):
    alpha = params[0]
    beta = params[1]
    fs = params[2]
    phi = params[3]
    return phi/((flux/fs)**alpha+(flux/fs)**beta + 1e-12)    


def generate_XD_model_dictionary(param_directory, tag1="glim24", tag2="", K_i = [2,2,2,2,3,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1]):
    """
    Construct a Python dictionary of model parameters for all the relevant classes.
    
    param_directory: directory where the files with expected formats are to be found.
    tag1, tag2: are used to name the files.
    K_i: Number of components for each class of objects.
    dNDm_type: Type of function to use to model g-mag distribution. 
        0 corresponds to power law and 1 to broken power law.
    """
    # Create empty dictionary
    params = {}
    
    # Adding dNdm parameters for each class
    for i in range(7):
        if dNdm_type[i] == 0:
            dNdm_params =np.loadtxt((param_directory+"%d-fit-pow-"+tag1)%i)
        else:
            dNdm_params =np.loadtxt((param_directory+"%d-fit-broken-"+tag1)%i)
        params[(i, "dNdm")] = dNdm_params
        
    # Adding GMM parameters for each class
    for i in range(7):
        amp, mean, covar = load_params_XD(param_directory,i,K_i[i],tag0="fit",tag1=tag1,tag2=tag2)
        params[(i,"amp")] = amp
        params[(i,"mean")] = mean
        params[(i,"covar")] = covar
        
    return params

def load_params_XD(param_directory,i,K,tag0="fit",tag1="glim24",tag2=""):
    fname = (param_directory+"%d-params-"+tag0+"-amps-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    amp = np.load(fname)
    fname = (param_directory+"%d-params-"+tag0+"-means-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    mean= np.load(fname)
    fname = (param_directory+"%d-params-"+tag0+"-covars-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    covar  = np.load(fname)
    return amp, mean, covar



def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def generate_XD_selection(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=[1., 1., 0., 0.25, 0., 0.25, 0.], \
                          reg_r=5e-4,zaxis="g", w_cc = 0.025, w_mag = 0.05, minmag = 21., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1]):
    """
    Summary: 
        - Set up a grid in the selection design region with the given grid parameters. 
            The grid is a rec array and has the following columns:
            gr, rz, mag, n_i (i in cnames), n_good, n_tot, FoM, select. The meaning of these values should be obvious.
        - For each cell, compute n_i, n_good, n_tot, FoM using the input parameters.
        - Rank order the cells and indicate whether each cell is included in the selection or not.
        - r is regularization parameter for FoM.

    Parameters:
        param_directory: Directory where model parameters are saved
        glim, rlim, zlim: 5-sigma detection limiting magnitudes. 
        gr_ref, rz_ref: Number density conserving global error reference point.
        reg_r: Regularization parameter. Empirically set to avoid pathologic 
            behaviors of the selection boundary.
        f_i: Various class weights for FoM.
        gmin, gmax: Minimum and maximum g-magnitude range to consider.
        K_i and dNdm_type: See generate_XD_model_dictionary() doc string.
        w_mag: Width in magnitude direction. 
        w_cc: Width in color-color grid.

    """

    params = generate_XD_model_dictionary(param_directory, K_i=K_i, dNdm_type=dNdm_type)

    # Create the grid.
    grid = generate_grid(w_cc, w_mag, minmag, maxmag)
    
    # cell volume
    Vcell = w_cc**2 * w_mag
    
    # Extracting the cell centers.
    gr = grid["gr"][:]
    rz = grid["rz"][:]
    mag = grid["mag"][:]
    flux = mag2flux(mag)

    # Compute the global error noise corresponding to each objects.
    const = 2.5/(5*np.log(10)) 
    gvar = (const * 10**(0.4*(mag-glim)))**2
    rvar = (const * 10**(0.4*(mag-gr_ref-rlim)))**2
    zvar = (const * 10**(0.4*(mag-gr_ref-rz_ref-zlim)))**2            
    
    # Compute the densities.
    for i in range(7):
        cname = cnames[i]
        grid[cname][:] = GMM_vectorized(rz, gr, params[i, "amp"], params[i, "mean"],params[i, "covar"], gvar, rvar, zvar) * dNdm(params[(i,"dNdm")], flux) * Vcell
        grid["Total"][:] += grid[cname][:] # Computing the total
        grid["FoM_num"][:] += f_i[i]*grid[cname][:]# Computing FoM_num
    
    # Computing FoM
    grid["FoM"][:] = grid["FoM_num"][:]/(grid["Total"][:]+reg_r+1e-12)
    
    # Rank order the cells according to FoM number.
    grid.sort(order='FoM')
    grid[:] = grid[::-1]
    
    # Selecting cells until the desired number N_tot is reached.
    N = 0
    last_FoM = 0
    for i in range(mag.size):
        if (N < N_tot):
            N += grid["Total"][i]
            grid["select"][i] = 1
            last_FoM = grid["FoM"][i]
        else:
            break               
    
    return grid, last_FoM    

def generate_XD_selection_var(param_directory, glim=23.8, rlim=23.4, zlim=22.4, \
                         glim_var=23.8, rlim_var=23.4, zlim_var=22.4,\
                          gr_ref=0.5, rz_ref=0.5, N_tot=2400, f_i=[1., 1., 0., 0.25, 0., 0.25, 0.], \
                          reg_r=5e-4,zaxis="g", w_cc = 0.025, w_mag = 0.05, minmag = 21., \
                          maxmag = 24., K_i = [2,2,2,3,2,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1]):
    """
    Summary: 
        - Set up a grid in the selection design region with the given grid parameters. 
            The grid is a rec array and has the following columns:
            gr, rz, mag, n_i (i in cnames), n_good, n_tot, FoM, select. The meaning of these values should be obvious.
        - For each cell, compute n_i, n_good, n_tot, FoM using the input parameters (glim, rlim, zlim).
        - Rank order the cells and indicate whether each cell is included in the selection or not.
        - r is regularization parameter for FoM.
        - After the selection is determined, re-compute the densities using glim_var, rlim_var, zlim_var

    Parameters:
        param_directory: Directory where model parameters are saved
        glim, rlim, zlim: Assumed 5-sigma detection limiting magnitudes.
        glim_var, rlim_var, zlim_var: "True" detection limits.
        gr_ref, rz_ref: Number density conserving global error reference point.
        reg_r: Regularization parameter. Empirically set to avoid pathologic 
            behaviors of the selection boundary.
        f_i: Various class weights for FoM.
        gmin, gmax: Minimum and maximum g-magnitude range to consider.
        K_i and dNdm_type: See generate_XD_model_dictionary() doc string.
        w_mag: Width in magnitude direction. 
        w_cc: Width in color-color grid.
    """
    
    params = generate_XD_model_dictionary(param_directory, K_i=K_i, dNdm_type=dNdm_type)

    # Create the grid.
    grid = generate_grid_var(w_cc, w_mag, minmag, maxmag)
    
    # cell volume
    Vcell = w_cc**2 * w_mag
    
    # --- Finding selection based on the assumed depth --- #
    # Extracting the cell centers.
    gr = grid["gr"][:]
    rz = grid["rz"][:]
    mag = grid["mag"][:]
    flux = mag2flux(mag)

    # Compute the global error noise corresponding to each objects.
    const = 2.5/(5*np.log(10)) 
    gvar = (const * 10**(0.4*(mag-glim)))**2
    rvar = (const * 10**(0.4*(mag-gr_ref-rlim)))**2
    zvar = (const * 10**(0.4*(mag-gr_ref-rz_ref-zlim)))**2            
    
    # Compute the densities.
    for i in range(7):
        cname = cnames[i]
        grid[cname][:] = GMM_vectorized(rz, gr, params[i, "amp"], params[i, "mean"],params[i, "covar"], gvar, rvar, zvar) * dNdm(params[(i,"dNdm")], flux) * Vcell
        grid["Total"][:] += grid[cname][:] # Computing the total
        grid["FoM_num"][:] += f_i[i]*grid[cname][:]# Computing FoM_num
    
    # Computing FoM
    grid["FoM"][:] = grid["FoM_num"][:]/(grid["Total"][:]+reg_r+1e-12)
    
    # Rank order the cells according to FoM number.
    grid.sort(order='FoM')
    grid[:] = grid[::-1]
    
    # Selecting cells until the desired number N_tot is reached.
    N = 0
    last_FoM = 0
    for i in range(mag.size):
        if (N < N_tot):
            N += grid["Total"][i]
            grid["select"][i] = 1
            last_FoM = grid["FoM"][i]
        else:
            break

    # --- Recomputing the densities based on the true depths --- #
    # Extracting the cell centers.
    gr = grid["gr"][:]
    rz = grid["rz"][:]
    mag = grid["mag"][:]
    flux = mag2flux(mag)

    # Compute the global error noise corresponding to each cell.
    const = 2.5/(5*np.log(10)) 
    gvar = (const * 10**(0.4*(mag-glim_var)))**2
    rvar = (const * 10**(0.4*(mag-gr_ref-rlim_var)))**2
    zvar = (const * 10**(0.4*(mag-gr_ref-rz_ref-zlim_var)))**2            
    
    # Compute the densities.
    grid["Total"][:] = 0
    grid["FoM_num"][:] = 0
    for i in range(7):
        cname = cnames[i]
        grid[cname][:] = GMM_vectorized(rz, gr, params[i, "amp"], params[i, "mean"],params[i, "covar"], gvar, rvar, zvar) * dNdm(params[(i,"dNdm")], flux) * Vcell
        grid["Total"][:] += grid[cname][:] # Computing the total
        grid["FoM_num"][:] += f_i[i]*grid[cname][:]# Computing FoM_num
    
    # Computing FoM
    grid["FoM"][:] = grid["FoM_num"][:]/(grid["Total"][:]+reg_r+1e-12)
    
    # Rank order the cells according to FoM number.
    grid.sort(order='FoM')
    grid[:] = grid[::-1]
    
    # Selecting cells until the desired number N_tot is reached.
    N = 0
    last_FoM_var = 0
    for i in range(mag.size):
        if (N < N_tot):
            N += grid["Total"][i]
            grid["select_var"][i] = 1
            last_FoM_var = grid["FoM"][i]
        else:
            break    
    
    return grid, last_FoM, last_FoM_var

def generate_grid_var(w_cc, w_mag, minmag, maxmag):
    """
    The same as generate_grid() except one additinoal column
    for selection boolean vector.
    """
    # Global params.
    xmin1,xmax1 = (-1.0,0.20)
    ymin1,ymax1 = (-.50,2.5)
    xmin2,xmax2 = (0.2,1.2)
    ymin2,ymax2 = (0.0,2.5)
    zmin,zmax = (minmag,maxmag)

    # +w*0.5 to center. Also note the convention [start, end)
    x1 = np.arange(xmin1,xmax1 + w_cc*0.9, w_cc)  + w_cc*0.5
    y1 = np.arange(ymin1,ymax1 + w_cc*0.9, w_cc ) + w_cc*0.5
    z1 = np.arange(zmin,zmax + w_mag*0.9, w_mag) + w_mag*0.5
    xv1, yv1,zv1 = np.meshgrid(x1, y1,z1)
    nx1,ny1,nz1 = (x1.size, y1.size, z1.size)
    xv1, yv1, zv1 = np.transpose(np.transpose(np.asarray([xv1,yv1,zv1])).reshape((nx1*ny1*nz1,3)))
    x2 = np.arange(xmin2+ w_cc*0.9,xmax2 + w_cc*0.9, w_cc)  + w_cc*0.5
    y2 = np.arange(ymin2+ w_cc*0.9,ymax2 + w_cc*0.9, w_cc ) + w_cc*0.5
    z2 = np.arange(zmin,zmax + w_mag*0.9, w_mag) + w_mag*0.5
    xv2, yv2,zv2 = np.meshgrid(x2, y2,z2)
    nx2,ny2,nz2 = (x2.size, y2.size, z2.size)
    xv2, yv2, zv2 = np.transpose(np.transpose(np.asarray([xv2,yv2,zv2])).reshape((nx2*ny2*nz2,3)))

#         print("Total number of cells: %d " % (nx1*ny1*nz1+nx2*ny2*nz2))

    xv =np.concatenate((xv1,xv2))
    yv = np.concatenate((yv1,yv2))
    zv = np.concatenate((zv1,zv2))

    grid = np.rec.fromarrays((xv, yv, zv,\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size,dtype=bool),np.zeros(xv.size,dtype=bool)), \
      dtype=[('gr','f8'),('rz','f8'), ('mag', 'f8'),\
             ('Gold','f8'),('Silver','f8'),('LowOII','f8'),\
             ('NoOII','f8'),('LowZ','f8'),('NoZ','f8'),\
             ('D2reject','f8'), ('FoM_num','f8'),('Total','f8'),\
             ('FoM','f8'), ('select','i4'),('select_var','i4')]);

    return grid    


def generate_grid(w_cc, w_mag, minmag, maxmag):
    # Global params.
    xmin1,xmax1 = (-1.0,0.20)
    ymin1,ymax1 = (-.50,2.5)
    xmin2,xmax2 = (0.2,1.2)
    ymin2,ymax2 = (0.0,2.5)
    zmin,zmax = (minmag,maxmag)

    # +w*0.5 to center. Also note the convention [start, end)
    x1 = np.arange(xmin1,xmax1 + w_cc*0.9, w_cc)  + w_cc*0.5
    y1 = np.arange(ymin1,ymax1 + w_cc*0.9, w_cc ) + w_cc*0.5
    z1 = np.arange(zmin,zmax + w_mag*0.9, w_mag) + w_mag*0.5
    xv1, yv1,zv1 = np.meshgrid(x1, y1,z1)
    nx1,ny1,nz1 = (x1.size, y1.size, z1.size)
    xv1, yv1, zv1 = np.transpose(np.transpose(np.asarray([xv1,yv1,zv1])).reshape((nx1*ny1*nz1,3)))
    x2 = np.arange(xmin2+ w_cc*0.9,xmax2 + w_cc*0.9, w_cc)  + w_cc*0.5
    y2 = np.arange(ymin2+ w_cc*0.9,ymax2 + w_cc*0.9, w_cc ) + w_cc*0.5
    z2 = np.arange(zmin,zmax + w_mag*0.9, w_mag) + w_mag*0.5
    xv2, yv2,zv2 = np.meshgrid(x2, y2,z2)
    nx2,ny2,nz2 = (x2.size, y2.size, z2.size)
    xv2, yv2, zv2 = np.transpose(np.transpose(np.asarray([xv2,yv2,zv2])).reshape((nx2*ny2*nz2,3)))

#         print("Total number of cells: %d " % (nx1*ny1*nz1+nx2*ny2*nz2))

    xv =np.concatenate((xv1,xv2))
    yv = np.concatenate((yv1,yv2))
    zv = np.concatenate((zv1,zv2))

    grid = np.rec.fromarrays((xv, yv, zv,\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size),np.zeros(xv.size),\
       np.zeros(xv.size),np.zeros(xv.size,dtype=bool)), \
      dtype=[('gr','f8'),('rz','f8'), ('mag', 'f8'),\
             ('Gold','f8'),('Silver','f8'),('LowOII','f8'),\
             ('NoOII','f8'),('LowZ','f8'),('NoZ','f8'),\
             ('D2reject','f8'), ('FoM_num','f8'),('Total','f8'),\
             ('FoM','f8'), ('select','i4')]);

    return grid    




def plot_slice(grid, m, bnd_fig_directory, fname="", movie_tag=None):
    """
    Given the projected grid, magnitude m, and the directory address, create a figure of boundary
    at the given slice.

    movie_tag: Must be an integer. Used to index slices that together can turn into a movie.
    """
    # Cell size
    cell_size=5.
    
    # Extract fields that are needed.
    gr = grid["gr"][:]
    rz = grid["rz"][:]
    mag = grid["mag"][:]
    iselect = grid["select"][:]==1

    # Picking out the right cells.
    mags = np.unique(grid["mag"][:])
    m_cell = mags[closest_idx(mags,m)]# The nearest center value 
#     print(mag.size,  (np.abs(mag-m_cell)<w_mag*0.6).sum())
    imag = mag == m_cell
    
    # Plotting
    # Accept region
    ibool = iselect & imag # Only the selected cells with mag == m_cell            
    plt.scatter(rz[ibool], gr[ibool], edgecolors="none", s=cell_size, c="green", alpha= 0.7)

    # Boundaries
    bnd_lw =2
    # FDR boundary:
    plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="red")
    plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="red")
    plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="red")
    plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="red") 

    # Decoration
    # Figure ranges
    plt.ylabel("$g-r$",fontsize=18)
    plt.xlabel("$r-z$",fontsize=18)
    plt.axis("equal")
    plt.axis([-0.5, 2.0, -0.5, 1.5])    

    plt.title("mag = %.3f" % m_cell, fontsize=15)

    # Save 
    if movie_tag is not None: # Then generate images with proper numbering for making a movie.
        plt.savefig((bnd_fig_directory+fname+"-mag%d-%d.png"%(0*1000,movie_tag)), bbox_inches="tight", dpi=400)
    else:
        plt.savefig((bnd_fig_directory+fname+"-mag%d.png"%(m_cell*1000)), bbox_inches="tight", dpi=400)            
    plt.close()

    return

def plot_slice_compare(grid, grid_ref, m, bnd_fig_directory, fname="", movie_tag=None):
    """
    Given the projected grid, grid2, magnitude m, and the directory address, create a figure of changes
    at the given slice m. Note that the grid must have the same registrations.

    color scheme:
    - Green: Regions where both old and new boundaries overlap.
    - Blue: Regions where the new boundary exclude the old.
    - Red: Regions where the new boundary is excluded where the old isn't.

    movie_tag: Must be an integer. Used to index slices that together can turn into a movie.
    """
    # Cell size
    cell_size=5.
    
    # Extract fields that are needed.
    # new grid
    gr = grid["gr"][:]
    rz = grid["rz"][:]
    mag = grid["mag"][:]
    iselect = grid["select"][:]==1

    # old grid
    gr2 = grid_ref["gr"][:]
    rz2 = grid_ref["rz"][:]
    mag2 = grid_ref["mag"][:]
    iselect2 = grid_ref["select"][:]==1    

    # Picking out the right cells.
    mags = np.unique(grid["mag"][:])
    m_cell = mags[closest_idx(mags,m)]# The nearest center value 
    imag = mag == m_cell
    imag2 = mag2 == m_cell

    # Redefine variables based on the selection.
    ibool = iselect & imag # Only the selected cells with mag == m_cell            
    gr = gr[ibool]
    rz = rz[ibool]
    ibool2 = iselect2 & imag2
    gr2 = gr2[ibool2]
    rz2 = rz2[ibool2]

    # Calculating venn diagram
    iAND12, i1NOT2, i2NOT1 = find_floating_point_venn_diagram(gr,rz, gr2, rz2)    

    # Plotting

    # In new but not old
    plt.scatter(rz[i1NOT2], gr[i1NOT2], edgecolors="none", s=cell_size, c="blue", alpha= 1.)
    # In old but not new
    plt.scatter(rz2[i2NOT1], gr2[i2NOT1], edgecolors="none", s=cell_size, c="red", alpha= 1.)
    # Intersection region
    plt.scatter(rz[iAND12], gr[iAND12], edgecolors="none", s=cell_size, c="green", alpha=1.)

    # Boundaries
    bnd_lw =2
    # FDR boundary:
    plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="red")
    plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="red")
    plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="red")
    plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="red") 

    # Decoration
    # Figure ranges
    plt.ylabel("$g-r$",fontsize=18)
    plt.xlabel("$r-z$",fontsize=18)
    plt.axis("equal")
    plt.axis([-0.5, 2.0, -0.5, 1.5])    

    plt.title("mag = %.3f" % m_cell, fontsize=15)

    # Save 
    if movie_tag is not None: # Then generate images with proper numbering for making a movie.
        plt.savefig((bnd_fig_directory+fname+"-mag%d-%d.png"%(0*1000,movie_tag)), bbox_inches="tight", dpi=400)
    else:
        plt.savefig((bnd_fig_directory+fname+"-mag%d.png"%(m_cell*1000)), bbox_inches="tight", dpi=400)            
    plt.close()

    return    

def closest_idx(arr, val):
    return np.argmin(np.abs(arr-val))

@nb.jit(nopython=True)
def find_floating_point_venn_diagram(x1, y1, x2, y2):
    """
    Given two 2D point sets, find the Venn diagram. 
    """
    iAND12 = np.zeros(x1.size, dtype=nb.boolean)
    i1NOT2 = np.ones(x1.size, dtype=nb.boolean)
    i2NOT1 = np.ones(x2.size, dtype=nb.boolean)

    for i in range(x1.size):
        ex, ey = x1[i], y1[i]
        AND_tmp = False
        for j in range(x2.size):
            if (np.abs(ex-x2[j])<1e-8) & (np.abs(ey-y2[j])<1e-8):
                iAND12[i] = True
                i2NOT1[j] = False
                i1NOT2[i] = False                
                break

    return iAND12, i1NOT2, i2NOT1


def plot_dNdm_XD(grid, grid2=None, fname=None, plot_type="DESI", glim=23.8, rlim=23.4, zlim=22.4,\
                glim2 = None, rlim2 =None, zlim2 = None, label1="", label2="Fid.", label3=None,\
                class_eff = [1., 1., 0., 0.6, 0., 0.25, 0.], 
                class_eff2 = [1., 1., 0., 0.6, 0., 0.25, 0.], lw=1.5):

    ibool = grid["select"][:]==1 # only interested in the selected cells.
    gmag = grid["mag"][:][ibool]
    rmag = gmag-grid["gr"][:][ibool]
    zmag = rmag-grid["rz"][:][ibool]
    if plot_type == "DESI":
        dNdm = np.zeros_like(gmag)
        for i in range(7):
            dNdm += class_eff[i]*grid[cnames[i]][:][ibool]
    elif plot_type == "Total":
        dNdm = grid["Total"][:][ibool]
    plt.hist(gmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="green", label="$g $ "+label1, lw=lw)
    plt.hist(rmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="red", label= "$r $ "+label1, lw=lw)
    plt.hist(zmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="purple",label="$z $ "+label1, lw=lw)
    
    if grid2 is not None:
        ibool = grid2["select"][:]==1 # only interested in the selected cells.
        gmag = grid2["mag"][:][ibool]
        rmag = gmag-grid2["gr"][:][ibool]
        zmag = rmag-grid2["rz"][:][ibool]
        if plot_type == "DESI":
            dNdm = np.zeros_like(gmag)
            for i in range(7):
                dNdm += class_eff2[i]*grid2[cnames[i]][:][ibool]
        elif plot_type == "Total":
            dNdm = grid2["Total"][:][ibool]

        plt.hist(gmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,color="green", alpha=0.25, histtype="stepfilled", label="$g $ "+label2, lw=lw)
        plt.hist(rmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,color="red", alpha=0.25, histtype="stepfilled", label="$r $ "+label2, lw=lw)
        plt.hist(zmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,color="purple", alpha=0.25, histtype="stepfilled", label="$z $ "+label2, lw=lw)

    if label3 is not None:
        ibool = grid["select_var"][:]==1 # only interested in the selected cells.
        gmag = grid["mag"][:][ibool]
        rmag = gmag-grid["gr"][:][ibool]
        zmag = rmag-grid["rz"][:][ibool]
        if plot_type == "DESI":
            dNdm = np.zeros_like(gmag)
            for i in range(7):
                dNdm += class_eff[i]*grid[cnames[i]][:][ibool]
        elif plot_type == "Total":
            dNdm = grid["Total"][:][ibool]
#         plt.hist(gmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="green", label="$g$ "+label3, lw=lw)
#         plt.hist(rmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="red", label= "$r$ "+label3, lw=lw)
#         plt.hist(zmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm,histtype="step",color="purple",label="$z$ "+label3, lw=lw)        
        ghist, edges = np.histogram(gmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm)
        rhist, _ = np.histogram(rmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm) # color="red", label= "$r$ "+label3, lw=lw)
        zhist, _ = np.histogram(zmag, bins = np.arange(20, 24.5, 0.025), weights=dNdm) # color="purple",label="$z$ "+label3, lw=lw)        
        centers = (edges[1:]+edges[:-1])/2.
        pt_size=5
        plt.scatter(centers, ghist, color="green", label="$g $ "+label3, marker="*",s=pt_size)
        plt.scatter(centers, rhist, color="red", label="$r $ "+label3, marker="*",s=pt_size)
        plt.scatter(centers, zhist, color="purple", label="$z $ "+label3, marker="*",s=pt_size)

    plt.axvline(glim,c="green", linestyle="--", lw=lw*1.5)
    plt.axvline(rlim,c="red", linestyle="--",  lw=lw*1.5)
    plt.axvline(zlim,c="purple", linestyle="--",  lw=lw*1.5)
    if (glim2 is not None):
    	if (np.abs(glim2-glim)>1e-6):
        	plt.axvline(glim2,c="green", linestyle="-.", lw=lw*1.5)
    if (rlim2 is not None):
    	if (np.abs(rlim2-rlim)>1e-6):
        	plt.axvline(rlim2,c="red", linestyle="-.", lw=lw*1.5)
    if (zlim2 is not None):
    	if (np.abs(zlim2-zlim)>1e-6):
        	plt.axvline(zlim2,c="purple", linestyle="-.", lw=lw*1.5)    
    plt.xlabel("Magnitude")
    plt.ylabel("Number density per 0.025 mag bin")
    plt.legend(loc="upper left")
    plt.xlim([20,24.5])
    plt.ylim([0,80])      

    plt.savefig(fname+".pdf", bbox_inches="tight", dpi=200)
    # plt.show()
    plt.close()