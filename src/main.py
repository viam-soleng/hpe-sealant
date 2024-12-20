import asyncio
import base64
import sys
from typing import Any, ClassVar, Final, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.media.video import ViamImage, CameraMimeType
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes
from viam.components.camera import CameraClient
from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image
from viam.errors import ViamError

import cv2
import numpy as np
from PIL import Image
import json


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
            result.image, contours = self.process_image(cam_image)
            # TODO: Add the contours to the extra field
            # result.extra = {"contours": json.dumps(contours[0])}
        else:
            raise ViamError(
                f"Requested camera {camera_name} is not a valid CameraClient"
            )
        return result

        # if self.camera.name != camera_name:
        #     raise ValueError("Camera not added as dependency")
        # cam_image = await self.camera.get_image()
        # result = CaptureAllResult()
        # result.image = self.process_image(cam_image)
        # return result

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
            return []  # await self.get_detections(image)
        else:
            raise ValueError(f"Camera {camera_name} is not a Camera resource")

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # TODO: Implement detection logic
        # result = self.process_image(image)
        return []

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
        # img = self.camera.get_image()
        # self.prepare_image(img)

        result = {"felix": "test"}
        return result

    def process_image(self, cam_image: ViamImage) -> tuple[ViamImage, List[Detection]]:
        # Convert the ViamImage to a PIL image
        pil_image = viam_to_pil_image(cam_image)
        # Convert the PIL image to a NumPy array
        np_image = np.array(pil_image)
        # Convert RGB to BGR (OpenCV uses BGR by default)
        if np_image.ndim == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        # Create a copy of the image to draw the contours on
        result_image = np_image.copy()
        # Convert the NumPy array to a grayscale image
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        # Threshold the image to create a binary image (black and white)
        _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)
        cv2.imwrite("bw_image.jpg", bw_image)
        wb_image = cv2.bitwise_not(bw_image)
        cv2.imwrite("wb_image.jpg", wb_image)
        # Find contours in the binary image
        contours, _ = cv2.findContours(wb_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours by area size (To be tweaked based upon the ideal shape)
        contours_filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (
                area < result_image.shape[0] * result_image.shape[1] * 0.4
                and area > result_image.shape[0] * result_image.shape[1] * 0.15
            ):
                # Keep only contours within a certain range
                contours_filtered.append(contour)
        self.logger.info(f"Number of contours: {len(contours_filtered)}")
        self.logger.info(f"Contour: {str(contours_filtered)}")
        # Convert the contours to a list of Detection objects
        # Display the filtered contours on top of the color image
        cv2.drawContours(result_image, contours_filtered, -1, (0, 255, 0), 3)
        # Convert the NumPy array back to a PIL image
        pil_image = matlike_to_pil(result_image)
        # Convert the PIL image to a ViamImage
        result_image = pil_to_viam_image(pil_image, CameraMimeType.JPEG)
        # return the processed image and a serialized version of the contours
        detections = [Detection]
        for contour in contours_filtered:
            detection = Detection()
            # TODO add bounding box
            # detection.bounding_box = cv2.boundingRect(contour)
            detection.class_name = json.dumps(contour_to_json(contour))
            detections.append(detection)
        return result_image, detections


def matlike_to_pil(np_image: np.ndarray) -> Image.Image:
    # np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    # Convert BGR to RGB (OpenCV uses BGR by default)
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(np_image)
    return pil_image


def contour_to_json(contour: np.ndarray) -> Mapping[str, Any]:
    # check if contour.dtype is a valid numpy dtype
    dtype = str(contour.dtype)
    shape = str(contour.shape)
    points = json.dumps(contour.tolist())
    contour_map = {"dtype": dtype, "shape": shape, "points": points}
    return contour_map


def jsoncontours_to_contours(json_contour: Mapping[str, Any]) -> List[np.ndarray]:
    points = json.loads(json_contour["points"])
    contour = np.array(points, dtype=json_contour["dtype"]).reshape(
        json_contour["shape"]
    )
    return contour


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
