#Author: Lilianne Nakazono

import numpy as np

def calculate_distance(table_seg:np.ndarray, radius:float=-1):
    """Calculates euclidian distance between the transient and the closest masked source (in pixels)

    Args:
        table_seg (np.ndarray): stamp of the segmented image
        radius (float, optional): radius of the search. Defaults to -1, which will be set to the size of the stamp.  

    Returns:
        int: x position of the closest masked source
        int: y position of the closest masked source
        float: euclidian distance between the transient and the closest masked source
    """

    # Converting the stamp to a binary mask
    table_seg[table_seg>0]=1

    # Defining the position of the transient
    transient_idx = 30 #assuming the size of the stamp is always ~60x60

    # Initializing variables
    breaker = False
    mask_i = None
    mask_j = None
    distance = 100 #out-of-range value

    # Initializing lists
    list_distance = []
    list_mask_i = []
    list_mask_j = []

    # Setting the radius of the search to the size of the stamp if not defined
    if radius == -1: 
        radius = transient_idx
    
    # If the transient is masked, the distance is 0
    if table_seg[transient_idx][transient_idx]==1:
        return transient_idx, transient_idx, 0
    
    # Searching for the closest masked source
    for step in np.arange(0, radius+1):
        array = np.arange(transient_idx-1-step,transient_idx+2+step)
        for indx, i  in enumerate(array):
            if indx==0 or i==int(transient_idx+1+step):
                for j in array:
                    if table_seg[i][j]==1:
                        list_distance.append(np.sqrt((transient_idx-i)**2+(transient_idx-j)**2))
                        list_mask_i.append(i)
                        list_mask_j.append(j)

            else:
                for j in [array[0],array[-1]]:
                    if table_seg[i][j]==1:
                        list_distance.append(np.sqrt((transient_idx-i)**2+(transient_idx-j)**2))
                        list_mask_i.append(i)
                        list_mask_j.append(j)

            if len(list_distance)>0:
                breaker = True
                break
        if breaker:
            break          

    # Getting the closest masked source
    index = np.where(np.array(list_distance)==np.min(list_distance))[0][0]
    mask_i = list_mask_i[index]
    mask_j = list_mask_j[index]
    distance = list_distance[index]

    return mask_i, mask_j, distance