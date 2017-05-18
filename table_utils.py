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
        knows how to compute. The final entry is how the user wants to format the end. For example, 
        "\\\\ \\hline"
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
    for e in return_format[:-1]:
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
    output_str = " & ".join(output_str)            
    output_str+=return_format[-1]

    return output_str

         

    
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
