import numpy as np
from scipy.stats import multivariate_normal

def summed_gm(pos, mus_fit, covar_fit, amps_fit):
    """
    Return a value evaluated with Gaussian mixture given means, covariances and relative amplitudes (that sum up to one).
    """
    func_val = 0.
    for i in range(amps_fit.size):
        func_val += amps_fit[i]*multivariate_normal.pdf(pos,mean=mus_fit[i], cov=covar_fit[i])    
    return func_val

def confidence_level_height_estimator(InitialGuess,confidenceVolume,Xrange, Yrange, Amps, Covs, Means, gridspacing=1e-2,tol =0.005, InitialLearningStep=0.1,NumGeneration=1e7,MaxIter=100):
    """
    For 2D gaussian mixture distribution, given the confidence level (say the top 10% region),
    returns a probability density value at which the "volume" within the corresponding contour is equal 
    to the given percentage input.
    Require: 
    - Parameters for the gaussian mixtures. Amps, Cov, Means. Amps should sum up to 1 if not it will be normalized.
    - ConfidenceVolume: Dafault is 50%. Measures the volume inside the contour.
    - NumGeneration: Number of points: Determines the number of points to generate to make estimations. Default is 1e9
    - InitialGuess: provides initigal guess for the density level corresponding to ConfidenceVolume. 1d array.
    - ConfidenceVolume: provides the confidence volumes for which to calculate.
    - InitialLearningStep: Tells by how much to change P initilally.
    - MaxIter: Maximum iteration for estimating the correct density level. If maxed out, then it returns an error sign.
    - tol: Tolerance of convergence of the confidence interval
    - gridspacing: 1e-4
    - Xrange, Yrange: Provide the limits.
    Output: The density level for contour plotting.
    Strategy: Amps define the multinomial distribution for the k gaussians. 
    Generates a number for the gaussian to use. 
    Generate a pair of random numbers (x,y) from the corresponding gaussian distribution. 
    Collect them in one dimensional array for X, Y.
    Once all the points are generated, generate two dimensional historgram. 
    Begin with the initial cut and estimate the volume within the corresponding contour by counting. 
    If it's too much volume inside, then raise the value by 
    the initial learning step. Continue to raise the value until there is too little volume inside. Update the learning step by 
    dividing it by two. Start decreasing the density level until there is too much volume inside. 
    Repeat this step until one reaches convergence or MaxIter is reached. Output the density level.
    
    Tips:
    - NumGeneration: More than 10e7 points might be too many. 
    - gridspacing: More than 1e-3 or more than 1e8 bins for the 2d histogram might be too many.
    
    Example
    -------
    # 2d example.
    Covs = np.array([[[1,0],[0,1]]])
    Means = np.array([[0,0]])
    Amps = np.array([1.])
    Xrange = [-5,5]
    Yrange = [-5,5]
    InitialGuess = 0.1
    gridspacing = 1e-2
    tol = 0.001

    # (2,1): Formatting
    il = 0.01 # Initial learning step
    returnP =  confidence_level_height_estimator([0.1,0.1,0.1,0.1,0.1],[0.05, 0.1, 0.2,0.5, 0.9],Xrange, Yrange, Amps, Covs, Means, gridspacing=1e-2,tol =0.005, InitialLearningStep=0.05,NumGeneration=1e7,MaxIter=100)

    returnP = returnP[::-1]

    Covs = np.array([[[1,0],[0,1]]])
    Means = np.array([[0,0]])
    Amps = np.array([1.])
    x = np.arange(-3.0,3.0,0.005)
    y = np.arange(-3.0,3.0,0.005)
    X,Y = np.meshgrid(x, y) # grid of point
    Z = summed_gm(np.transpose(np.array([Y,X])), Means, Covs, Amps) # evaluation of the function on the grid
    plt.contour(X,Y,Z,returnP,linewidths=2,cmap="RdBu")
    plt.axis("equal")
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.show()
    plt.close()    
    
    """
    # Defining the multinomilal distribution
    sourceNumGeneration = np.random.multinomial(NumGeneration, Amps, size=1)[0]

    # Generating the random numbers from corresponding gaussains and collecting them into one dimensional X,Y vectors.
    sampleX = np.array([])
    sampleY = np.array([])
    for i in range(Amps.size):
        # For each gaussian i, generate sourceNumGenreation[i] samples. Add to sample X,Y
        array = np.random.multivariate_normal(mean=Means[i] ,cov=Covs[i], size=sourceNumGeneration[i])
        tempX = array[:,0]
        tempY = array[:,1]
        sampleX = np.concatenate((sampleX,tempX))
        sampleY = np.concatenate((sampleY,tempY))
    # Construct the 2d histogram
    xedges = np.arange(Xrange[0],Xrange[1],gridspacing)
    yedges = np.arange(Yrange[0],Yrange[1],gridspacing)
    H, _, _ = np.histogram2d(sampleX, sampleY, bins=(xedges, yedges),normed=1.)
    # Probability integral within a density contour is given by H[H>p*]*gridspacing**2
    
    returnP = []
    for j in range(len(confidenceVolume)):
        guessP = InitialGuess[j]
        cv = confidenceVolume[j]
        learningStep = InitialLearningStep
        iteration = 0 
        while (abs(np.sum(H[H>guessP])*gridspacing**2-cv)>tol) & (iteration<MaxIter):
            if ((np.sum(H[H>guessP])*gridspacing**2-cv)>0)& (iteration<MaxIter):
                while ((np.sum(H[H>guessP])*gridspacing**2-cv)>0)& (iteration<MaxIter):
                    guessP += learningStep
                    iteration +=1
    #                 print("guess, learning step, current volume", guessP, learningStep, (np.sum(H[H>guessP])*gridspacing**2))            
                else:
                    learningStep = learningStep/2
                    iteration +=1
    #                 print("guess, learning step, current volume", guessP, learningStep, np.sum(H[H>guessP])*gridspacing**2)            
            else:
                while ((np.sum(H[H>guessP])*gridspacing**2-cv)<0)& (iteration<MaxIter):
                    guessP -= learningStep
                    iteration +=1
    #                 print("guess, learning step, current volume", guessP, learningStep,np.sum(H[H>guessP])*gridspacing**2 )            
                else:
                    learningStep = learningStep/2
                    iteration +=1
    #                 print("guess, learning step, current volume", guessP, learningStep,np.sum(H[H>guessP])*gridspacing**2 )
        if (iteration<MaxIter)&(guessP>0):
            returnP.append(guessP)
        else:
            returnP.append(0)
        
    return returnP






import numpy as np
from scipy.stats import multivariate_normal

def summed_gm(pos, mus_fit, covar_fit, amps_fit):
    """
    Return a value evaluated with Gaussian mixture given means, covariances and relative amplitudes (that sum up to one).
    """
    func_val = 0.
    for i in range(amps_fit.size):
        func_val += amps_fit[i]*multivariate_normal.pdf(pos,mean=mus_fit[i], cov=covar_fit[i])    
    return func_val

def inverse_cdf_gm(cvs,Xrange, Yrange, Amps, Covs, Means, gridspacing=1e-3, gridnumber = 1e2):
    """Given parameters for a gaussian mixture and a confidence interval volume, 
    returns the corresponding probability density level.
    
    Requires:
    - Amps, Means, Covs: Parametesr for a gaussian mixture
    - cvs: Confidence volume(s) in a list format.
    - grid-spacing: Grid-spacing to try
    - Xrange, Yrange: Ranges in which to compute the guassians.
    
    Return:
    - probability density lvel
    
    Example:
    --------
    """
    x = np.arange(Xrange[0],Xrange[1],gridspacing)
    y = np.arange(Yrange[0],Yrange[1],gridspacing)
    
    X,Y = np.meshgrid(x,y)
    Z = summed_gm(np.transpose(np.array([Y,X])), Means, Covs, Amps) # evaluation of the function on the grid
    pdf_sorted = np.sort(Z.flatten())

    # Prob grid 
    prob_grid = np.arange(0, pdf_sorted[-1],pdf_sorted[-1]/1e3)
    
    # Calculating cumulative function (that is probability volume within the boundary)
    cdf = np.zeros(prob_grid.size)
    for i in range(cdf.size):
        cdf[i] = np.sum(pdf_sorted[pdf_sorted>prob_grid[i]])*gridspacing**2
    
    pReturn = []
    for i in range(len(cvs)):
        cv = cvs[i]
        pReturn.append(prob_grid[find_nearest_idx(cdf,cv)])

    return pReturn
        
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

        
    
    
