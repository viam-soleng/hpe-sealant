from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple
import cv2
from cv2.typing import MatLike
import numpy as np
import pickle
from viam.media.video import ViamImage
from viam.proto.service.vision import Classification, Detection
from PIL import Image


def find_contours(image: Image) -> Tuple[List[MatLike], List[Detection]]:
    """This function finds contours in an image."""
    np_image = pil_to_opencv(image)
    # Convert RGB to BGR gray scale (OpenCV uses BGR by default)
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    # Threshold the image to create a binary image (black and white)
    # Create thresholded B/W image using Otsu's method
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)
    # Invert the binary image (black becomes white and vice versa)
    wb_image = cv2.bitwise_not(bw_image)
    # Find the contours in the image
    contours_all, _ = cv2.findContours(wb_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area size (To be tweaked based upon the ideal shape)
    # TODO: Expose filter parameters as configuration
    contours_filtered: Sequence[MatLike] = []
    detections: List[Detection] = []
    for idx, contour in enumerate(contours_all):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (
            area < np_image.shape[0] * np_image.shape[1] * 0.4
            and area > np_image.shape[0] * np_image.shape[1] * 0.15
            and h < wb_image.shape[0]
            and w < wb_image.shape[1]
        ):
            # Keep only contours within a certain range
            contours_filtered.append(contour)
            detection = Detection(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
            detection.confidence = 1.0
            detection.class_name = str(len(contours_filtered) - 1)
            detections.append(detection)
    return (contours_filtered, detections)


def contour_to_dict(contour: np.ndarray) -> Mapping[str, Any]:
    dtype = str(contour.dtype)
    shape = tuple(
        int(dim) for dim in contour.shape
    )  # Ensure shape dimensions are integers
    points = contour.tolist()
    contour_map = {"dtype": dtype, "shape": shape, "data": points}
    return contour_map


def draw_contours(
    image: Image,
    contours: List[np.ndarray],
    color: Optional[Tuple[int, int, int]] = None,
) -> Image:
    """This function draws the contours on the image.

    Args:
        image (np.ndarray): The image to draw the contours on
        contours (List[np.ndarray]): The contours to draw

    Returns:
        np.ndarray: The image with the contours drawn on it
    """
    if color is None:
        color = (0, 255, 0)
    image = pil_to_opencv(image)
    cv2.drawContours(image, contours, -1, color, 3)
    image_with_contours = opencv_to_pil(image)
    return image_with_contours


def save_contours(contour: np.ndarray, filename: str) -> None:
    """This function saves the contour to a file.

    Args:
        contour (np.ndarray): The contour to save
        filename (str): The filename to save the contour to
    """
    with open(filename, "wb") as f:
        pickle.dump(contour, f)


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
