import numpy as np

cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]


def class_breakdown_cut(cn, weight, area,rwd="D", num_classes=8, \
     return_format= ["cut", "rwd" ,"type", "Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched",
      "DESI", "Total", "Eff", "FoM"],\
     class_eff = [1., 1., 0.25, 0.25, 0., 0., 0., 0.]
     ):
    """
    Given class number, weights, and areas, return the breakdown of object 
    for each class. 

    return_format: A list that shows which information should be included in the returned string.
        For each element in the array, if the object is a string in cnames, then the corresponding
        classes counts is printed. Also, DESI, Total and Eff are special strings that the function
        knows how to compute.
    class_eff: Short for class efficiency. Assign expected yield to Gold through DR3unmatched objects.
    """
    
    # Computing counts
    if rwd == "R":
        counts = generate_raw_breakdown(cn)[:num_classes]
    elif rwd == "W":
        counts = generate_weighted_breakdown(cn, weight, num_classes)
    else:
        counts = generate_density_breakdown(cn, weight, area, num_classes)

    Total = np.sum(counts)
    DESI = np.sum(np.asarray(class_eff)*counts)
    eff = DESI/Total

    output_str = []
    for e in return_format:
        if e in cnames:
            idx = cnames.index(e)
            output_str.append("%d"%counts[idx])
        elif e == "Total":
            output_str.append("%d"%Total)
        elif e == "DESI":
            output_str.append("%d"%DESI)
        elif e == "Eff":
            output_str.append("%.3f"%eff)
        elif e == "rwd":
            output_str.append(rwd)
        else:
            output_str.append(e)

    return " & ".join(output_str)

def plot_dNdz_selection(cn, w, iselect1, redz, area_total, dz=0.05,  gold_eff=1, silver_eff=1, NoZ_eff=0.25, NoOII_eff=0.25,\
     iselect2=None, plot_total=True, fname="dNdz.png", color1="black", color2="red", color_total="green",\
     label1="Selection 1", label2="Selection 2", label_total="DEEP2 Total", wNoOII=0.2, wNoZ=0.5):
    """
    Given class number (cn), mask (iselect1), weights (w), redshifts, class efficiencies, plot the redshift
    histogram. 

    dz: Histogram binwidth
    **_eff: Gold and Silver are always equal to one. NoZ and NoOII are objects wtih no redshift
        in DEEP2 but are guessed to have efficiency of about 0.25.
    iselect2: If not None, used as another set of mask to plot dNdz histogram.
    plot_total: Plots total.
    fname: Saves in fname.
    color1: iselect1 color
    color2: iselect2 color
    color_total: total color
    label1, label2, lbael_total: Labels
    """

    if plot_total:
        ibool = np.logical_or((cn==0),(cn==1)) 
        plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w[ibool]/area,\
                 histtype="step", color=color_total, label=label_total)

        # NoOII:
        ibool = (cn==3) 
        N_NoOII = NoOII_eff*w[ibool].sum();
        plt.bar(left=0.6, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoOII (Proj.)", hatch="*")
        # NoZ:
        ibool = (cn==5) 
        N_NoZ = NoZ_eff*w[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoZ (Proj.)")


    if iselect2 is not None:
        ibool = np.logical_or((cn==0),(cn==1)) & iselect2
        plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w[ibool]/area,\
                 histtype="step", color=color2, label=label2)

        # NoOII:
        ibool = (cn==3) & iselect2
        N_NoOII = NoOII_eff*w[ibool].sum();
        plt.bar(left=0.6, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+ " NoOII (Proj.)", hatch="*")
        # NoZ:
        ibool = (cn==5) & iselect2
        N_NoZ = NoZ_eff*w[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+" NoZ (Proj.)")         


    # Selection 1.
    ibool = np.logical_or((cn==0),(cn==1)) & iselect1 # Total
    plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w[ibool]/area,\
             histtype="step", color=color1, label=label1)

    # NoOII:
    ibool = (cn==3) & iselect1
    N_NoOII = NoOII_eff*w[ibool].sum();
    plt.bar(left=0.6, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color1, \
            edgecolor =color1, label=label1+" NoOII (Proj.)", hatch="*")
    # NoZ:
    ibool = (cn==5) & iselect1
    N_NoZ = NoZ_eff*w[ibool].sum();
    plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color1, \
            edgecolor =color1, label=label1+" NoZ (Proj.)")


    plt.xlim([0.5,1.4+wNoZ+0.1])
    plt.legend(loc="upper right", fontsize=15)  
    ymax=250
    if plot_total:
        ymax = 450
    plt.ylim([0,ymax])
    # plt.legend(loc="upper left")
    plt.xlabel("Redshift z")
    plt.ylabel("dN/d(%.3fz) per sq. degs."%dz)
    plt.savefig(fname, bbox_inches="tight", dpi=400)
    # plt.show()
    plt.close()    


         

    
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
    return np.bincount(cn.astype(int))

def generate_weighted_breakdown(cn, weight,num_classes):
    counts = np.zeros(num_classes,dtype=int)
    for i in range(num_classes):
        if (i<6):
            counts[i] = np.sum(weight[cn==i])
        else:
            counts[i] = np.sum(cn==i)
    return counts

def generate_density_breakdown(cn, weight,area,num_classes):
    counts = np.zeros(num_classes,dtype=int)
    for i in range(num_classes):
        if (i<6):
            counts[i] = np.sum(weight[cn==i])/np.float(area)
        else:
            counts[i] = np.sum(cn==i)/np.float(area)
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
    return ["Gold", "Silver", "LowOII","NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched" ,"D2unobserved"]
