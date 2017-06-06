# Load modules
import numpy as np
from xd_elg_utils import *
from astropy.io import fits as FITS
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

%matplotlib inline
# Constants
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]


# Global variables
MMT_data_dir = "./MMT_data/"
photo_data_fname = "DR1-MMT-DR3-23hr.fits"
spec_data_fname = "spHect-23hrs_1.1933-0100.fits"
title_header = "23hr"
panel_dir = "./panel/23hr/"
threshold=5
pix_window_size = 151
width_guesses = np.arange(0.5,10,0.1)
xmax = 8250
xmin = 4500
##############################################################################

# Load targets
print("Load target photo info.")
table = load_fits_table(photo_data_fname) 
fib_idx_observed = table["FIB_NUM"][table["OBSERVED"]==1]-1 # -1 for zero indexing    



# Load spectra
print("Load MMT spec data from spHect-file. Only our targets.")
x, d, divar, AND_mask = load_MMT_specdata(spec_data_fname)#, fib_idx_observed)


# Inspect x spacing
del_x = (x[:,1:]-x[:,:-1]).flatten()
# plt.hist(del_x,np.arange(0.5,1.5,0.001),histtype="step")
# plt.show()
# plt.close()
print("Grid spacing stats:")
x_median = np.median(del_x)
x_std =  np.std(del_x)
print("Mean/std: %.3f/%.3f"%(x_median,x_std))



idx_example = 154 # Spec number 155
x_example, d_example, AND_mask_example, divar_example = x[idx_example], d[idx_example], AND_mask[idx_example], divar[idx_example]

# AND_mask condition 
mask = AND_mask_example>0

## Take boxed car average
# d_example_boxed = box_car_avg(d_example,mask=mask,window_pixel_size=pix_window_size)
# d_example_boxed_subtracted = d_example - d_example_boxed

## Savgol approach
# polyorder=3
# d_savgol = savgol_filter(d_example, pix_window_size, polyorder, mode='constant')
# plot_spectrum(x_example,d_example, x_example, d_savgol, mask=mask)

# Median approach
d_median = median_filter(d_example, mask=mask, window_pixel_size=pix_window_size)
# plot_spectrum(x_example,d_example, x_example, d_median, mask=mask)

# Plot spec after median subtraction
d_example_median_subtracted = d_example-d_median
# plot_spectrum(x_example,d_example_median_subtracted, mask=mask)

# Given the boxed spectrum, compute A, varA, and chi sq
print("Single guess")
width_guess = 3
A, varA, chi, S2N = process_spec(d_example_median_subtracted, divar_example, width_guess, x_median, mask=mask)
plot_fit(x_example, d_example, A, S2N, chi, mask=mask, mask_caution=None, xmin=4500, \
         xmax=8250, s=20, plot_show=True, plot_save=False, plot_title="")

# print("Multiple guesses, pick best")
# A, varA, chi, S2N = process_spec_best(d_example_median_subtracted, divar_example, width_guesses, x_median, mask=mask)
# plot_fit(x_example, d_example,A, S2N, chi, mask=mask, mask_caution=None, xmin=4500, \
#          xmax=8250, s=10, plot_show=True, plot_save=False, plot_title="")




# Load spectra
print("Load MMT spec data from spHect-file. Only our targets.")
x, d, divar, AND_mask = load_MMT_specdata(spec_data_fname, fib_idx_observed)




# Pre-allocate memory for A, S2N
A_array = np.zeros_like(x)
S2N_array = np.zeros_like(x)
AND_array = np.zeros(x.shape, dtype=bool)
Chi_array = np.zeros_like(x)

# For each, spectrum process them as save their A and S2N 
for i in range(A_array.shape[0]):
# for i in range(5):
    if (i%10)==0:
        print(i)
    # Get mask
    AND_tmp = AND_mask[i,:]
    mask = AND_tmp>0
    AND_array[i,:] = mask
    
    # Get ivar
    divar_tmp = divar[i,:]
    
    # Get continuum subtracted spectrum
    d_tmp = d[i,:]
    d_median = median_filter(d_tmp, mask=mask, window_pixel_size=pix_window_size)
    d_median_subtracted = d_tmp-d_median
    
    # Get the processed data.
#     A_array[i,:], _, Chi_array[i,:], S2N_array[i,:] = process_spec_best(d_median_subtracted, divar_tmp, width_guesses, x_median, mask=mask)
    A_array[i,:], _, Chi_array[i,:], S2N_array[i,:] = process_spec(d_median_subtracted, divar_tmp, 3, x_median , mask=mask)    




# Find where high S2N recurs frequently
S2N_flat = S2N_array.flatten()
mask = AND_array.flatten() & (S2N_flat>threshold)
x_flat = x.flatten()[mask]

dx = 5
hist, bin_edges = np.histogram(x_flat, bins=np.arange(xmin, xmax, dx))
bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.

print((hist>1).sum()/((xmax-xmin)/dx))

x_bad = bin_centers[hist>1]
fig = plt.figure(figsize=(15,5))
plt.scatter(bin_centers, hist, edgecolor="none", color="blue",s=5)
plt.xlim([xmin, xmax])
plt.show()
plt.close()



for i in range(x.shape[0]):
#     if (i%10)==0:
#         print(i)
    S2N_tmp = S2N_array[i]
    A_tmp = A_array[i]
    chi_tmp = Chi_array[i]
    x_tmp = x[i]
    AND_tmp = AND_mask[i]
    d_tmp = d[i]

    mask = AND_tmp>0
    
    mask_caution = np.zeros(x_tmp.size,dtype=bool)
    for xb in x_bad:
        mask_caution |= np.logical_and((x_tmp>(xb-dx/2.)),(x_tmp<(xb+dx/2.)))
    
    ibool = np.logical_not(mask) & (x_tmp>xmin) & (x_tmp<xmax)
    if (S2N_tmp[ibool]>threshold).sum()>0:
        print(i)
        title_str = "-".join([("%d"%i),title_header,str(fib_idx_observed[i]+1)])
        plot_fit(x_tmp, d_tmp, A_tmp, S2N_tmp, chi_tmp, mask=mask, mask_caution=mask_caution, xmin=xmin, \
                 xmax=xmax, s=20, plot_show=False, plot_save=True, save_dir=panel_dir, plot_title=title_str)   