import os
import matplotlib.pyplot as plt
import numpy as np
from calculate_distance import calculate_distance

def check_path(file_path:str):
    """Checks if the output file exists

    Args:
        file_path (str): path to the file

    Returns:
        bool: True if the file exists, False otherwise
    """
    if os.path.exists(file_path):
        return True
    else:
        return False
    
def plot_distance(objid:str, science_img:np.ndarray=None, template_img:np.ndarray=None, original_img: np.ndarray=None, replace:bool=False, save:bool=True, output_path:str=None):
    """ Plot science masked image, template masked image and the distance between the transient and the closest masked source

    Args:
        objid (str): Object identification from ZTF
        science_img (np.ndarray, optional): 2-d array for the science masked image. Defaults to None.
        template_img (np.ndarray, optional): 2-d array for the template masked image. Defaults to None.
        original_img (np.ndarray, optional): 2-d array for the original science image. Defaults to None.
        replace (bool, optional): if true, replace figure if file already exists . Defaults to False.
        save (bool, optional): If true, save figure. Defaults to True.
        output_path (str, optional): Path to the output folder. Defaults to None.
    """
    
    # Check if the output path exists
    if save == True:
        if (check_path(output_path+objid+".png") or check_path(output_path+objid+"_hostless.png")):
            if replace:
                pass
            else:
                return
    
    # Plot the figure
    fig, ax = plt.subplots(1,3,figsize=(15,5))

    # Plot the science image
    i, j, distance_sci = calculate_distance(science_img)
    if i is not None:
        ax[0].plot(j,i, "x", color="yellow")

    ax[0].imshow(science_img, origin="lower", cmap="RdGy_r")
    ax[0].text(2,2,"Distance: %.2f px" % (distance_sci), color="yellow")
    ax[0].plot(30,30, ".", color="yellow")
    circle1 = plt.Circle((30, 30), 7, color='yellow', fill=False)
    ax[0].add_patch(circle1)
    ax[0].set_title("Science")

    # Plot the distance between the transient and the closest masked source
    i, j, distance_temp = calculate_distance(template_img)
    if i is not None:
        ax[1].plot(j,i, "x", color="yellow")

    ax[1].imshow(template_img, origin="lower", cmap="RdGy_r")
    ax[1].text(2,2,"Distance: %.2f px" % (distance_temp), color="yellow")
    ax[1].plot(30,30, ".", color="yellow")
    circle1 = plt.Circle((30, 30), 7, color='yellow', fill=False)
    ax[1].add_patch(circle1)
    ax[1].set_title("Template")

    # Plot the original image
    ax[2].imshow(original_img, origin="lower", cmap="hot", norm="log")
    ax[2].set_title("Science Cutout")

    # Set the title
    fig.suptitle(objid, fontsize=20)

    # Set the output path
    output_path = output_path+objid+".png"
    plt.ioff()

    # Save the figure
    if save:
        plt.savefig(output_path)
    else:
        plt.show()

    return fig
