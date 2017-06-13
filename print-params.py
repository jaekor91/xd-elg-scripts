import numpy as np
from xd_elg_utils import *
import XD_selection_module as XD
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]


param_directory = "./XD-parameters/"
K_i = [2,2,2,3,2,2,7]
dNdm_type = [1, 1, 0, 1, 0, 0, 1]
param_tag2 = "-Field34" # "-Field34" if using only Field 3 and 4 data. Otherwise "". 

##############################################################################
print("Parameter directory: ")
if param_tag2 == "-Field34":
	print("Using Field 3 and 4 training fits.")



##############################################################################
print("Load the parameters")
params = XD.generate_XD_model_dictionary(param_directory, K_i=K_i, dNdm_type=dNdm_type, tag2=param_tag2)
print("Completed\n")

##############################################################################
# Class name, CN, component gaussian, A_ij, mu_ij_rz [1], mu_ij_gr [0], rzrz, rzgr, grgr
# Class Name $i$ & Component Gaussian $j$ & $A_{ij}$& $\mu^{g-r}_{ij}$ & $ \mu^{r-z}_{ij}$ & $\Sigma^{g-r,g-r}_{ij} $ & $ \Sigma^{g-r,r-z}_{ij}$ & $\Sigma^{r-z,r-z}_{ij}$\\ \hline
print("Print GMM fit parameters")
num_class = 7
for i in range(num_class):
    idx = np.flip(np.argsort(params[(i,"amp")]), -1)
    counter = 0
    for j in idx:
        counter +=1
#         print(i,j)
        row_str = []
        
        # Class name
        if counter == 1:
            row_str.append(cnames[i])
        else:
            row_str.append("")
            
        # Class number.
        # Nothing
        
        # Component gaussian
        row_str.append("%d"%counter)
        
        # A_ij
        row_str.append("%.6f"%params[(i,"amp")][j])

        # mu_ij_rz [0]
        row_str.append("%.6f"%params[(i,"mean")][j][0])        
        
        # mu_ij_gr [1]
        row_str.append("%.6f"%params[(i,"mean")][j][1])                

        # Covar
        # rzrz
        row_str.append("%.6f"%params[(i,"covar")][j][0, 0])
        # rzgr
        row_str.append("%.6f"%params[(i,"covar")][j][0, 1])                
        # grgr
        row_str.append("%.6f"%params[(i,"covar")][j][1, 1])
        
        row_str = " & ".join(row_str)
        row_str += "\\\\ \\hline"
        
        print(row_str)
print("Completed\n")


##############################################################################
print("dNdM fit parameters")
# Class Name, Type, alpha, beta, f, phi
num_class = 7
for i in range(num_class):
    row_str = []

    # Class name
    row_str.append(cnames[i])
        
    # By type
    if dNdm_type[i]==0:
        row_str.append("Power")
        # Alpha
        row_str.append("%.5f"%(-params[(i, "dNdm")][0]))
        # Beta
        row_str.append("---")        
        # Fs
        row_str.append("%d"%1)        
        # phi
        row_str.append("%.5f"%params[(i, "dNdm")][1])
    else:
        row_str.append("Broken")
        # Alpha
        row_str.append("%.5f"%(params[(i, "dNdm")][0]))
        # Beta
        row_str.append("%.5f"%(params[(i, "dNdm")][1]))
        # Fs
        row_str.append("%.5f"%(params[(i, "dNdm")][2]))
        # phi
        row_str.append("%.1f"%params[(i, "dNdm")][3])        
    
    # 
    row_str = " & ".join(row_str)
    row_str += "\\\\ \\hline"

    print(row_str)
print("Completed\n")
