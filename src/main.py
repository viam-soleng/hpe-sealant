import asyncio
import pickle
import os
from typing import Any, ClassVar, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.media.video import ViamImage, CameraMimeType
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

from .contours import (
    find_contours,
    load_contours,
    save_contours,
    draw_contours,
    contour_to_dict,
    contour_to_detection,
    compare_hausdorff,
    ViamContour,
)


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

        # Check if draw_contours is set to string in config
        if "draw_contours" in config.attributes.fields:
            if not config.attributes.fields["draw_contours"].HasField("string_value"):
                raise Exception("draw_contours must be a string.")
            draw_contours = config.attributes.fields["draw_contours"].string_value
            # Check if draw_contours is not set to reference or detected or both
            if not draw_contours in ["reference", "detected", "both"]:
                raise Exception(
                    "draw_contours must be set to reference, detected or both"
                )
        # Check if max_contours is set to number in config
        if "max_contours" in config.attributes.fields:
            if not config.attributes.fields["max_contours"].HasField("number_value"):
                raise Exception("max_contours must be a number.")
            max_contours = config.attributes.fields["max_contours"].number_value
            if not max_contours >= 0:
                raise Exception("max_contours must be a positive number.")

        # Check if min_area and max_area are set to number in config
        if "min_area" in config.attributes.fields:
            if not config.attributes.fields["min_area"].HasField("number_value"):
                raise Exception("min_area must be a number.")
            min_area = config.attributes.fields["min_area"].number_value
            if not min_area >= 0:
                raise Exception("min_area must be a positive number.")
        if "max_area" in config.attributes.fields:
            if not config.attributes.fields["max_area"].HasField("number_value"):
                raise Exception("max_area must be a number.")
            max_area = config.attributes.fields["max_area"].number_value
            if not max_area >= 0:
                raise Exception("max_area must be a positive number.")

        # Check if min_width and max_width are set to number in config
        if "min_width" in config.attributes.fields:
            if not config.attributes.fields["min_width"].HasField("number_value"):
                raise Exception("min_width must be a number.")
            min_width = config.attributes.fields["min_width"].number_value
            if not min_width >= 0:
                raise Exception("min_width must be a positive number.")
        if "max_width" in config.attributes.fields:
            if not config.attributes.fields["max_width"].HasField("number_value"):
                raise Exception("max_width must be a number.")
            max_width = config.attributes.fields["max_width"].number_value
            if not max_width >= 0:
                raise Exception("max_width must be a positive number.")

        # Check if min_height and max_height are set to number in config
        if "min_height" in config.attributes.fields:
            if not config.attributes.fields["min_height"].HasField("number_value"):
                raise Exception("min_height must be a number.")
            min_height = config.attributes.fields["min_height"].number_value
            if not min_height >= 0:
                raise Exception("min_height must be a positive number.")
        if "max_height" in config.attributes.fields:
            if not config.attributes.fields["max_height"].HasField("number_value"):
                raise Exception("max_height must be a number.")
            max_height = config.attributes.fields["max_height"].number_value
            if not max_height >= 0:
                raise Exception("max_height must be a positive number.")

        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        self.logger.info("Reconfiguring Sealant service")

        if "draw_contours" in config.attributes.fields:
            self.draw_contours = config.attributes.fields["draw_contours"].string_value
            if self.draw_contours in ["reference", "detected", "both"]:
                self.logger.info(f"Drawing {self.draw_contours} contours on the image")
        else:
            self.draw_contours = ""
            self.logger.info("Not drawing contours on the image")

        if "max_contours" in config.attributes.fields:
            self.max_contours = int(
                config.attributes.fields["max_contours"].number_value
            )
            self.logger.info(f"Max number of contours: {self.max_contours}")

        if "min_area" in config.attributes.fields:
            self.min_area = int(config.attributes.fields["min_area"].number_value)
            self.logger.info(f"Min area of contours: {self.min_area}")
        else:
            self.min_area = 0

        if "max_area" in config.attributes.fields:
            self.max_area = int(config.attributes.fields["max_area"].number_value)
            self.logger.info(f"Max area of contours: {self.max_area}")
        else:
            self.max_area = 0

        if "min_width" in config.attributes.fields:
            self.min_width = int(config.attributes.fields["min_width"].number_value)
            self.logger.info(f"Min width of contours: {self.min_width}")
        else:
            self.min_width = 0

        if "max_width" in config.attributes.fields:
            self.max_width = int(config.attributes.fields["max_width"].number_value)
            self.logger.info(f"Max width of contours: {self.max_width}")
        else:
            self.max_width = 0

        if "min_height" in config.attributes.fields:
            self.min_height = int(config.attributes.fields["min_height"].number_value)
            self.logger.info(f"Min height of contours: {self.min_height}")
        else:
            self.min_height = 0

        if "max_height" in config.attributes.fields:
            self.max_height = int(config.attributes.fields["max_height"].number_value)
            self.logger.info(f"Max height of contours: {self.max_height}")
        else:
            self.max_height = 0
        # Load the reference contours from the pickle file
        self.ref_contours = load_contours("contours.pickle")
        # Store the dependencies for later use
        self.dependencies = dependencies
        return

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
        result = CaptureAllResult(detections=[], extra={})
        contours: List[ViamContour] = []
        if isinstance(camera, CameraClient):
            image = await camera.get_image()
            pil_image = viam_to_pil_image(image)
            contours = find_contours(
                pil_image,
                min_area=self.min_area,
                max_area=self.max_area,
                max_contours=self.max_contours,
                min_width=self.min_width,
                max_width=self.max_width,
                min_height=self.min_height,
                max_height=self.max_height,
            )
            # TODO: Add the opencv contours to the extra field if needed
            # result.extra["cv_contours"] = contour_to_dict(res_contours)
            # Compare contours with reference contours
            if len(self.ref_contours) > 0 and len(contours) > 0:
                res_contours = compare_hausdorff(self.ref_contours, contours)
                # extract the hausdorff distances from the result list and add them to the extra field
                result.extra["contours"] = [
                    {
                        "area": ctr.area,
                        "hausdorff": ctr.hausdorff,
                        "arclength": ctr.arclenght,
                    }
                    for ctr in res_contours
                ]
            else:
                result.extra["contours"] = [
                    {
                        "area": ctr.area,
                        "arclength": ctr.arclenght,
                        "hausdorff": "no reference contour",
                    }
                    for ctr in contours
                ]
            # self.logger.info(f"# Reference Contours: {len(self.ref_contours)}")
            # self.logger.info(f"Reference Contours: \n{self.ref_contours}")
            # Draw the detected or reference contours on the image. Default is none.
            if (
                self.draw_contours == "detected" or self.draw_contours == "both"
            ) and len(contours) > 0:
                pil_image = draw_contours(pil_image, contours, (0, 0, 255))
                # Add the contours bounding boxes to the result.detections
                for det_idx, ctr in enumerate(contours):
                    det = ctr.detection
                    det.class_name = f"detected_{det_idx}"
                    result.detections.append(det)
            if (
                self.draw_contours == "reference" or self.draw_contours == "both"
            ) and len(self.ref_contours) > 0:
                pil_image = draw_contours(pil_image, self.ref_contours, (0, 255, 0))
                # Add the contours bounding boxes to the result.detections
                for ref_idx, ref_ctr in enumerate(self.ref_contours):
                    ref_det = ref_ctr.detection
                    ref_det.class_name = f"reference_{ref_idx}"
                    result.detections.append(ref_det)
            result.image = pil_to_viam_image(pil_image, image.mime_type)

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
        contours = find_contours(
            viam_to_pil_image(image),
            min_area=self.min_area,
            max_area=self.max_area,
            max_contours=self.max_contours,
            min_width=self.min_width,
            max_width=self.max_width,
            min_height=self.min_height,
            max_height=self.max_height,
        )
        detections: List[Detection] = []
        for ctr in contours:
            detections.append(ctr["detection"])
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
            self.logger.info("save_contours: %s", command)
            if "camera_name" not in command:
                raise ViamError("Missing camera_name in command")
            try:
                camera = self.dependencies[
                    CameraClient.get_resource_name(command["camera_name"])
                ]
            except KeyError:
                raise ViamError(
                    f"Requested camera {command["camera_name"]} is not listed in dependencies"
                )
            if isinstance(camera, CameraClient):
                image = await camera.get_image()
                pil_image = viam_to_pil_image(image)
                contours = find_contours(
                    pil_image,
                    min_area=self.min_area,
                    max_area=self.max_area,
                    max_contours=self.max_contours,
                    min_width=self.min_width,
                    max_width=self.max_width,
                    min_height=self.min_height,
                    max_height=self.max_height,
                )
                self.ref_contours = contours
                save_contours(contours, "contours.pickle")
                # pil_image = draw_contours(pil_image, contours)
                return {
                    "result": f"{len(contours)} contours saved to file and loaded as reference"
                }
            else:
                raise ViamError(
                    f"Requested camera {command["camera_name"]} is not a valid CameraClient"
                )
        if command["command"] == "delete_contours":
            self.logger.info("delete_contours: %s", command)
            if os.path.exists("contours.pickle"):
                os.remove("contours.pickle")
            else:
                self.logger.info("No reference contours file found")
            self.ref_contours = []
            return {"result": "Reference contours deleted"}
        raise ViamError(f"Unknown command {command}")


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
