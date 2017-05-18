import numpy as np



def class_breakdown_cut(cn, weight, area, rwd="D", num_classes=8):
    """
    Given class number, weights, and areas, return the breakdown of object 
    for each class.
    """
    
    # Computing counts
    if rwd == "R":
        counts = generate_raw_breakdown(cn)[:num_classes]
    elif rwd == "W":
        counts = generate_weighted_breakdown(cn, weight, num_classes)
    else:
        counts = generate_density_breakdown(cn, weight, area, num_classes)

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
