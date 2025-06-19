from PIL import Image
import cv2
import numpy as np


def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV image (NumPy array).

    Args:
        pil_image (Image.Image): The PIL image to convert.

    Returns:
        np.ndarray: The converted OpenCV image.
    """
    # Convert PIL image to NumPy array
    np_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV uses BGR by default)
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return np_image


def opencv_to_pil(np_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (NumPy array) to a PIL image.

    Args:
        np_image (np.ndarray): The OpenCV image to convert.

    Returns:
        Image.Image: The converted PIL image.
    """
    # Convert BGR to RGB (OpenCV uses BGR by default)
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(np_image)

    return pil_image
