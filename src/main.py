import asyncio
from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple


from typing_extensions import Self
from viam.media.video import ViamImage
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes
from viam.components.camera import CameraClient
from viam.media.utils.pil import viam_to_pil_image
from viam.errors import ViamError

import cv2
from cv2.typing import MatLike
import numpy as np


class Sealant(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hpe-automotive", "sealant-check"), "sealant"
    )

    dependencies: Mapping[ResourceName, ResourceBase]

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        # Store the dependencies for later use
        self.dependencies = dependencies
        return super().reconfigure(config, dependencies)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        try:
            camera = self.dependencies[CameraClient.get_resource_name(camera_name)]
        except KeyError:
            raise ViamError(
                f"Requested camera {camera_name} is not listed in dependencies"
            )
        result = CaptureAllResult()
        if isinstance(camera, CameraClient):
            cam_image = await camera.get_image()
            (contours, detections) = self.find_contours(cam_image)
            result.image = cam_image
            # Return the bounding boxes of the contours
            result.detections = detections
            # Return the contours as extra information
            result.extra = {
                "contours": [contour_to_dict(contour) for contour in contours]
            }
        else:
            raise ViamError(
                f"Requested camera {camera_name} is not a valid CameraClient"
            )
        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        try:
            camera = self.dependencies[CameraClient.get_resource_name(camera_name)]
        except KeyError:
            raise ViamError(
                f"Requested camera {camera_name} is not listed in dependencies"
            )
        if type(camera) == CameraClient:
            image = await camera.get_image()
            return await self.get_detections(image)
        else:
            raise ValueError(f"Camera {camera_name} is not a Camera resource")

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # Return the bounding boxes of the contours
        (_, detections) = self.find_contours(image)
        return detections

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def do_command(
        self, command, *, timeout=None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()

    def find_contours(
        self, cam_image: ViamImage
    ) -> Tuple[List[MatLike], List[Detection]]:
        # Convert the ViamImage to a PIL image
        pil_image = viam_to_pil_image(cam_image)
        # Convert the PIL image to a NumPy array
        np_image = np.array(pil_image)
        # Convert RGB to BGR (OpenCV uses BGR by default)
        if np_image.ndim == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        # Convert the NumPy array to a grayscale image
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        # Threshold the image to create a binary image (black and white)
        # The threshold value is determined using Otsu's method but might need to be tweaked:
        # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
        _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)
        # Invert the binary image (black becomes white and vice versa)
        wb_image = cv2.bitwise_not(bw_image)
        # Find the contours in the image
        contours_all, _ = cv2.findContours(
            wb_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter contours by area size (To be tweaked based upon the ideal shape)
        # TODO: Expose filter parameters as configuration
        contours_filtered: Sequence[MatLike] = []
        detections: List[Detection] = []
        for idx, contour in enumerate(contours_all):
            area = cv2.contourArea(contour)
            if (
                area < np_image.shape[0] * np_image.shape[1] * 0.4
                and area > np_image.shape[0] * np_image.shape[1] * 0.15
            ):
                # Keep only contours within a certain range
                contours_filtered.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                detection = Detection(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
                detection.confidence = 1.0
                detection.class_name = str(len(contours_filtered) - 1)
                detections.append(detection)
        self.logger.debug(f"Number of contours after filter: {len(contours_filtered)}")
        return (contours_filtered, detections)


def contour_to_dict(contour: np.ndarray) -> Mapping[str, Any]:
    dtype = str(contour.dtype)
    shape = tuple(
        int(dim) for dim in contour.shape
    )  # Ensure shape dimensions are integers
    points = contour.tolist()
    contour_map = {"dtype": dtype, "shape": shape, "data": points}
    return contour_map


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
