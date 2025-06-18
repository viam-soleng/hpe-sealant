from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import cv2
from cv2.typing import MatLike
import numpy as np
import pickle
from viam.proto.service.vision import Detection
from PIL import Image

from dataclasses import dataclass


@dataclass
class ViamContour:
    """Class for keeping track of contour information."""

    contour: Sequence[MatLike]
    area: float
    arclenght: Optional[float]
    width: int
    height: int
    hausdorff: Optional[Dict[str, float]]
    detection: Optional[Detection]


def find_contours(
    image: Image,
    min_area: int,
    max_area: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    max_contours: int,
    cfg_thresh: int = 0,
) -> Tuple[List[ViamContour], Image.Image]:
    """This function finds contours in an image and returns the contours and the thresholded image."""
    np_image = pil_to_opencv(image)
    bw_image = thresholding(np_image, cfg_thresh)
    # Find the contours in the image
    contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by width and height
    if min_width > 0 or min_height > 0 or max_width > 0 or max_height > 0:
        contours = filter_contours_by_width_height(
            contours, min_width, min_height, max_width, max_height
        )
    if min_area > 0 or max_area > 0:
        contours = filter_contours_by_area(contours, min_area, max_area)
    if max_contours > 0:
        contours = contours[:max_contours]
    viam_contours: List[ViamContour] = []
    for ctraw in contours:
        vctr = ViamContour(
            contour=ctraw,
            area=cv2.contourArea(ctraw),
            arclenght=cv2.arcLength(ctraw, True),
            width=cv2.boundingRect(ctraw)[2],
            height=cv2.boundingRect(ctraw)[3],
            detection=contour_to_detection(ctraw),
            hausdorff={},
        )
        viam_contours.append(vctr)
    return viam_contours, opencv_to_pil(cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB))


def thresholding(image: np.ndarray, cfg_thresh: int) -> np.ndarray:
    """This function applies a threshold to the image.

    Args:
        image (np.ndarray): The image to apply the threshold to
        cfg_thresh (int): The threshold value

    Returns:
        np.ndarray: The thresholded image
    """
    # Convert RGB to BGR gray scale (OpenCV uses BGR by default)
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Threshold the image to create a binary image (black and white)
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = otsu_thresh - cfg_thresh
    _, bw_image = cv2.threshold(blur, adjusted_thresh, 255, cv2.THRESH_BINARY)
    # Invert the binary image (black becomes white and vice versa)
    wb_image = cv2.bitwise_not(bw_image)
    return wb_image


def draw_contours(
    image: Image,
    contours: List[ViamContour],
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
    for contour in contours:
        cv2.drawContours(image, [contour.contour], -1, color, 3)
    image_with_contours = opencv_to_pil(image)
    return image_with_contours


def save_contours(contours: List[ViamContour], filename: str) -> None:
    """This function saves the contour to a file.

    Args:
        contour (ViamContour): The contour to save
        filename (str): The filename to save the contour to
    """
    for contour in contours:
        # Clear the detection field before saving as Detection is not serializable
        contour.detection = None
    with open(filename, "wb") as f:
        pickle.dump(contours, f)


def load_contours(filename: str) -> List[ViamContour]:
    """This function loads the contour from a file.

    Args:
        filename (str): The filename to load the contour from

    Returns:
        ViamContour: The loaded contour
    """
    contours: List[ViamContour] = []
    try:
        with open(filename, "rb") as f:
            contours: List[ViamContour] = pickle.load(f)
        for contour in contours:
            # Restore the detection field after loading
            contour.detection = contour_to_detection(contour.contour)
    except:
        pass
    return contours


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


def compare_hausdorff(
    ref_contours: List[ViamContour], det_contours: List[ViamContour]
) -> List[ViamContour]:
    """This function compares the reference contours with the detected contours using Hausdorff distance.
    This metric measures the maximum distance between any point on one contour and the closest point on the other contour.
    It is useful for assessing the overall dissimilarity between shapes, even if they have slight variations in form.

    Args:
        ref_contours (List[np.ndarray]): The reference contours
        det_contours (List[np.ndarray]): The detected contours
    """
    result: List[ViamContour] = []
    # Create Hausdorff distance extractor
    hausdorff_dist = cv2.createHausdorffDistanceExtractor()

    # Compute the Hausdorff distance between the reference and detected contours
    for det_idx, det_contour in enumerate(det_contours):
        for ref_idx, ref_contour in enumerate(ref_contours):
            # Compute the Hausdorff distance between the reference and detected contours
            distance = hausdorff_dist.computeDistance(
                ref_contour.contour, det_contour.contour
            )
            det_contour.hausdorff[str(ref_idx)] = distance
        result.append(det_contour)
    return result


def contour_to_detection(contour: MatLike) -> Detection:
    """Convert a contour to a Detection object.

    Args:
        contour (np.ndarray): The contour to convert.

    Returns:
        Detection: The converted Detection object.
    """
    x, y, w, h = cv2.boundingRect(contour)
    detection = Detection(
        x_min=x,
        y_min=y,
        x_max=x + w,
        y_max=y + h,
    )
    detection.confidence = 1.0
    detection.class_name = "contour_bbox"
    return detection


def filter_contours_by_area(
    contours: List[MatLike], min_area: int, max_area: int
) -> Sequence[MatLike]:
    """Filter contours by area size.

    Args:
        contours (List[np.ndarray]): The contours to filter.
        min_area (int): The minimum area size.
        max_area (int): The maximum area size.

    Returns:
        List[np.ndarray]: The filtered contours.
    """
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (min_area <= area <= max_area) or (min_area <= area and max_area == 0):
            filtered_contours.append(contour)
    return filtered_contours


def filter_contours_by_width_height(
    contours: Sequence[MatLike],
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
) -> Sequence[MatLike]:
    """Filter contours by width and height.

    Args:
        contours (List[np.ndarray]): The contours to filter.
        min_width (int): The minimum width.
        min_height (int): The minimum height.

    Returns:
        List[MatLike]: The filtered contours.
    """
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (min_width <= w <= max_width or min_width <= w and max_width == 0) and (
            min_height <= h <= max_height or min_height <= h and max_height == 0
        ):
            filtered_contours.append(contour)
    return filtered_contours


def contour_to_dict(contour: np.ndarray) -> Mapping[str, Any]:
    dtype = str(contour.dtype)
    shape = tuple(
        int(dim) for dim in contour.shape
    )  # Ensure shape dimensions are integers
    points = contour.tolist()
    contour_map = {"dtype": dtype, "shape": shape, "data": points}
    return contour_map


def contours_min_distance(image: Image.Image, c1: np.ndarray, c2: np.ndarray):
    """Find the minimum distance between two contours and return the closest points.
    Args:
        c1 (np.ndarray): The first contour.
        c2 (np.ndarray): The second contour.
    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], float]: The closest points and the minimum distance.
    """
    min_dist = max(image.width, image.height)
    chosen_point_c2 = None
    chosen_point_c1 = None
    for point in c1:
        t = point[0][0], point[0][1]
        index, dist = closest_point(t, c2[:, 0])
        if dist[index] < min_dist:
            min_dist = dist[index]
            chosen_point_c2 = tuple(c2[index][0])
            chosen_point_c1 = t
    return chosen_point_c1, chosen_point_c2, min_dist


def closest_point(point, array):
    diff = array - point
    distance = np.einsum("ij,ij->i", diff, diff)
    return np.argmin(distance), distance


def mark_defect(
    image: Image.Image,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> Image.Image:
    """
    Draws a circle to mark a defect on the given image between two points.

    This function calculates the center and radius of a circle defined by two points (p1 and p2),
    draws the circle on the image using the specified color, and returns the modified image.

    Args:
        image (Image.Image): The input PIL image on which to mark the defect.
        p1 (Tuple[int, int]): The first point (x, y) defining the defect location.
        p2 (Tuple[int, int]): The second point (x, y) defining the defect location.
        color (Tuple[int, int, int], optional): The color of the circle in BGR format. Defaults to (255, 0, 0).

    Returns:
        Image.Image: The PIL image with the defect marked as a circle.
    """
    center = (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2),
    )
    radius = int(np.linalg.norm(np.array(p1) - np.array(p2)))  # / 2)
    np_image = pil_to_opencv(image)  # BGR format!
    cv2.circle(np_image, center, radius + 20, (0, 255, 255), 2)
    cv2.circle(np_image, p1, radius, (0, 0, 255), 2)
    cv2.circle(np_image, p2, radius, (0, 0, 255), 2)
    return opencv_to_pil(np_image)
