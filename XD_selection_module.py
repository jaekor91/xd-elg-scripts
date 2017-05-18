import numpy as np

cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]

def apply_XD_globalerror(objs, last_FoM, param_directory, glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5,reg_r=1e-4/(0.025**2 * 0.05),f_i=[1., 1., 0., 0.25, 0., 0.25, 0.],\
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