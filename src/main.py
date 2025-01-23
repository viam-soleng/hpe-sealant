import asyncio
import pickle
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
from viam.errors import ViamError
from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image


from .contours import find_contours, save_contours, draw_contours, contour_to_dict


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
        # Load the reference contours from the pickle file
        # catch if no file is foun
        try:
            with open("contours.pickle", "rb") as f:
                self.ref_contours = pickle.load(f)
        except FileNotFoundError:
            self.ref_contours = None
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
            image = await camera.get_image()
            pil_image = viam_to_pil_image(image)
            (contours, detections) = find_contours(pil_image)
            # Draw the contours on the image if no reference contours are provided
            if self.ref_contours is None or len(self.ref_contours) == 0:
                result.image = pil_to_viam_image(
                    draw_contours(pil_image, contours), image.mime_type
                )
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
        (_, detections) = find_contours(image)
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
        if command["command"] == "save_contours":
            if "camera_name" not in command:
                raise ViamError("Missing camera_name in command")
            try:
                camera = self.dependencies[
                    CameraClient.get_resource_name({command["camera_name"]})
                ]
            except KeyError:
                raise ViamError(
                    f"Requested camera {command["camera_name"]} is not listed in dependencies"
                )
            if isinstance(camera, CameraClient):
                image = await camera.get_image()
                pil_image = viam_to_pil_image(image)
                (contours, _) = find_contours(pil_image)
                save_contours(contours, "filename")
                pil_image = draw_contours(pil_image, contours)
                return pil_to_viam_image(pil_image)
            else:
                raise ViamError(
                    f"Requested camera {command["camera_name"]} is not a valid CameraClient"
                )
        raise ViamError(f"Unknown command {command}")


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
