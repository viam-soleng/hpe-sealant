from __future__ import annotations
from typing import Self, TYPE_CHECKING, List, Tuple
from PIL import Image
import cv2
import numpy as np
from src.utils import distance_to_detection, opencv_to_pil, pil_to_opencv, mark_defect
from viam.proto.service.vision import Detection

if TYPE_CHECKING:
    from src.main import Sealant


def analyze_image(self: Sealant, image: Image) -> Tuple[Image.Image, List[Detection]]:
    min_distance = 0.0
    max_distance = 0.0
    detections: List[Detection] = []
    np_image = pil_to_opencv(image)
    thresh_image = threshold_image(np_image, self.thresh_offset)
    if self.bw_image:
        np_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
    filtered_contours = find_contours(self, thresh_image)
    for i, contour in enumerate(filtered_contours):
        area = cv2.contourArea(contour)
        self.logger.debug(f"Contour {i} area: {area}")
    if self.draw_contours:
        np_image = cv2.drawContours(np_image, filtered_contours, -1, (0, 255, 0), 1)
    if len(filtered_contours) == 2:
        p_min_1, p_min_2, min_distance, p_max_1, p_max_2, max_distance = (
            check_sealant_width(np_image, filtered_contours[0], filtered_contours[1])
        )
        if self.mark_detections:
            np_image = mark_defect(
                np_image,
                p_min_1,
                p_min_2,
            )
            np_image = mark_defect(
                np_image,
                p_max_1,
                p_max_2,
            )
        points = [p_min_1, p_min_2, p_max_1, p_max_2]
        if any(p is None for p in points):
            self.logger.warning(
                f"One or more points for sealant width analysis are None: "
                f"p_min_1={p_min_1}, p_min_2={p_min_2}, p_max_1={p_max_1}, p_max_2={p_max_2}"
            )
        detections.append(distance_to_detection(p_min_1, p_min_2, min_distance))
        detections.append(distance_to_detection(p_max_1, p_max_2, max_distance))
    else:
        self.logger.warning(
            "Expected exactly two contours for sealant width analysis, found: {}".format(
                len(filtered_contours)
            )
        )
    return opencv_to_pil(np_image), detections


def check_sealant_width(image: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    min_dist = max(image.shape[:2])
    chosen_min_point_c2 = None
    chosen_min_point_c1 = None

    max_dist = 0.0
    chosen_max_point_c2 = None
    chosen_max_point_c1 = None
    for point in c1:
        t = point[0][0], point[0][1]
        index, distance = closest_point(t, c2[:, 0])
        if distance < min_dist:
            min_dist = distance
            chosen_min_point_c2 = tuple(c2[index][0])
            chosen_min_point_c1 = t
        if distance > max_dist:
            max_dist = distance
            chosen_max_point_c2 = tuple(c2[index][0])
            chosen_max_point_c1 = t
    return (
        chosen_min_point_c1,
        chosen_min_point_c2,
        min_dist,
        chosen_max_point_c1,
        chosen_max_point_c2,
        max_dist,
    )


def closest_point(point, array):
    diff = array - point
    distances = np.einsum("ij,ij->i", diff, diff)
    idx = np.argmin(distances)
    return idx, np.sqrt(distances[idx])


def check_contour_anomalies(self: Self, filtered_contours: List) -> List:
    anomalies = []
    # If there are two reference contours, calculate the distance between them
    # contour1 = self.ref_contours[0]
    # contour2 = self.ref_contours[1]
    # distance = cv2.pointPolygonTest(contour1, tuple(contour2[0][0]), True)
    # analysis_results = {
    #    "distance": distance,
    #    "filtered_contours": filtered_contours,
    # }


def find_contours(self: Sealant, image: np.ndarray) -> List:
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on the configuration
    filtered_contours = filter_contours_by_size(
        contours,
        min_area=self.min_area,
        max_area=self.max_area,
        min_width=self.min_width,
        min_height=self.min_height,
        max_width=self.max_width,
        max_height=self.max_height,
        max_contours=self.max_contours,
    )
    return filtered_contours


def threshold_image(image: np.ndarray, cfg_thresh: int) -> np.ndarray:
    # Convert RGB to BGR gray scale (OpenCV uses BGR by default)
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Blur and then threshold the image to create a binary image (black and white)
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = otsu_thresh - cfg_thresh
    _, bw_image = cv2.threshold(blur, adjusted_thresh, 255, cv2.THRESH_BINARY)
    # Invert the binary image (black becomes white and vice versa)
    wb_image = cv2.bitwise_not(bw_image)
    return wb_image


def filter_contours_by_size(
    contours,
    min_area=100,
    max_area=10000,
    min_width=0,
    min_height=0,
    max_width=0,
    max_height=0,
    max_contours=0,
) -> list:
    filtered_contours = filter_contours_by_area(contours, min_area, max_area)
    filtered_contours = filter_contours_by_width_height(
        filtered_contours, min_width, min_height, max_width, max_height
    )
    if max_contours > 0:
        contours = contours[:max_contours]
    return filtered_contours


# Filter contours by area
def filter_contours_by_area(contours, min_area, max_area):
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if (min_area == 0 or area >= min_area) and (max_area == 0 or area <= max_area):
            filtered.append(c)
    return filtered


# Filter contours by width and height
def filter_contours_by_width_height(
    contours, min_width, min_height, max_width, max_height
):
    filtered = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (
            (min_width == 0 or w >= min_width)
            and (min_height == 0 or h >= min_height)
            and (max_width == 0 or w <= max_width)
            and (max_height == 0 or h <= max_height)
        ):
            filtered.append(c)
    return filtered


def filter_contours_by_shape(
    filtered_contours: list, ref_contours: list, threshold: float = 0.1
) -> list:
    """
    Finds and returns contours from a list that closely match any of the given reference contours invariant to scale!.

    This function compares each contour in `filtered_contours` to each contour in `ref_contours` using
    the cv2.matchShapes method. If the similarity measure is below the specified `threshold`, the contour
    is considered a match and added to the result list.

    Args:
        filtered_contours (list): List of contours to be checked for similarity.
        ref_contours (list): List of reference contours to compare against.
        threshold (float, optional): Maximum allowed shape difference for a match. Defaults to 0.1.

    Returns:
        list: Contours from `filtered_contours` that match any reference contour within the threshold.
    """
    matched_contours = []
    for contour in filtered_contours:
        for ref_contour in ref_contours:
            # Compare the contour with the reference contour
            match = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if match < threshold:
                matched_contours.append(contour)
                break
    return matched_contours
