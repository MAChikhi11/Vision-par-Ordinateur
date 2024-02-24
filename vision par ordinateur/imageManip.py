import math
import cv2

import numpy as np

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `cv2.imread()` function - 
          whatch out  for the returned color format ! Check the following link for some fun : 
          https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT

    # Utilisez cv2.imread - le format RGB doit être retourné
    out = cv2.imread(image_path)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    ### VOTRE CODE ICI - FIN

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float32) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = 0.5* np.square(image)
    ### VOTRE CODE ICI - FIN

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: see if you can use  the opencv function `cv2.cvtColor()` 
    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT    
    out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ### VOTRE CODE ICI - FIN

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    if channel == "R":
        out = image.copy()
        out[:, :, 0] = 0  # Set red channel to 0
    elif channel == "G":
        out = image.copy()
        out[:, :, 1] = 0  # Set green channel to 0
    elif channel == "B":
        out = image.copy()
        out[:, :, 2] = 0  # Set blue channel to 0
    else:
        # If the channel is not "R", "G", or "B", return the original image
        out = image

    ### VOTRE CODE ICI - FIN

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT

    # Conversion de l'image en espace LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

  # Extraction du canal spécifié
    if channel == "L":
     out = lab_image[...,0]

    elif channel == "A":
     out = lab_image[...,1]

    elif channel == "B":
     out = lab_image[...,2]

    else:
     raise ValueError(f"Canal invalide : {channel}")
    
    ### VOTRE CODE ICI - FIN
    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract the specified channel
    if channel == 'H':
        out = hsv_image[..., 0]
    elif channel == 'S':
        out = hsv_image[..., 1]
    elif channel == 'V':
        out = hsv_image[..., 2]
    else:
        raise ValueError("Invalid channel. Use 'H', 'S', or 'V'.")
    ### VOTRE CODE ICI - FIN

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### VOTRE CODE ICI - DEBUT
   # Get the left half of image1
    left_half_image1 = image1[:, :image1.shape[1] // 2, :]

    # Get the right half of image2
    right_half_image2 = image2[:, image2.shape[1] // 2:, :]

    # Exclude specified channels using rgb_exclusion
    excluded_channel_image1 = rgb_exclusion(left_half_image1, channel1)
    excluded_channel_image2 = rgb_exclusion(right_half_image2, channel2)

    # Concatenate the left half of image1 and the right half of image2
    out = np.concatenate((excluded_channel_image1, excluded_channel_image2), axis=1)
    ### VOTRE CODE ICI - FIN

    return out


def mix_quadrants(image):
    """
    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    
    # Dimensions de l'image
    hauteur, largeur, _ = image.shape

    # Quadrant supérieur gauche : Supprimer le canal 'R'
    quadrant_sup_gauche = rgb_exclusion(image[:hauteur // 2, :largeur // 2, :], 'R')

    # Quadrant supérieur droit : Assombrir l'image
    quadrant_sup_droit = dim_image(image[:hauteur // 2, largeur // 2:, :])

    # Quadrant inférieur gauche : Éclaircir l'image
    quadrant_inf_gauche = np.power(image[hauteur // 2:, :largeur // 2, :], 0.5)

    # Quadrant inférieur droit : Supprimer le canal 'R'
    quadrant_inf_droit = rgb_exclusion(image[hauteur // 2:, largeur // 2:, :], 'R')

    # Concaténer les 4 quadrants pour former l'image de sortie
    partie_superieure = np.concatenate((quadrant_sup_gauche, quadrant_sup_droit), axis=1)
    partie_inferieure = np.concatenate((quadrant_inf_gauche, quadrant_inf_droit), axis=1)
    out = np.concatenate((partie_superieure, partie_inferieure), axis=0)

    ### VOTRE CODE ICI - FIN

    return out
