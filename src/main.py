import asyncio
from typing import Any, ClassVar, Deque, List, Mapping, Optional, Sequence

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
from viam.utils import from_dm_from_extra
from viam.errors import NoCaptureToStoreError
from queue import Queue
from collections import deque
from typing import Dict
import uuid

from src.analyze import analyze_image


class Sealant(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hpe-automotive", "sealant-check"), "sealant"
    )

    dependencies: Mapping[ResourceName, ResourceBase]

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
            if not config.attributes.fields["draw_contours"].HasField("bool_value"):
                raise Exception("draw_contours must be a boolean.")

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
        if "thresh_offset" in config.attributes.fields:
            if not config.attributes.fields["thresh_offset"].HasField("number_value"):
                raise Exception("thresh_offset must be a number.")
            thresh_offset = config.attributes.fields["thresh_offset"].number_value
        if "bw_image" in config.attributes.fields:
            if not config.attributes.fields["bw_image"].HasField("bool_value"):
                raise Exception("bw_image must be a boolean.")
            bw_image = config.attributes.fields["bw_image"].bool_value
            if not isinstance(bw_image, bool):
                raise Exception("bw_image must be a boolean.")
        return []

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

        vs = cls(config.name)
        vs.reconfigure(config, dependencies)
        return vs

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        self.logger.info("Reconfiguring Sealant service")

        # Cache capture_all_from_camera results to be confirmed for upload
        self.capture_all_cache: Deque[Dict[str, CaptureAllResult]] = deque(maxlen=10)
        # Queue to store capture_all_from_camera results to be uploaded by data manager
        self.capture_all_queue: Queue = Queue()
        if "max_contours" in config.attributes.fields:
            self.max_contours = int(
                config.attributes.fields["max_contours"].number_value
            )
        else:
            self.max_contours = 0
        self.logger.info(f"Max number of contours: {self.max_contours}")

        if "draw_contours" in config.attributes.fields:
            self.draw_contours = config.attributes.fields["draw_contours"].bool_value
        else:
            self.draw_contours = False
        self.logger.info("Draw contours on the image: %s", self.draw_contours)

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

        if "thresh_offset" in config.attributes.fields:
            self.thresh_offset = int(
                config.attributes.fields["thresh_offset"].number_value
            )
        else:
            self.thresh_offset = 0
        self.logger.info(f"Adjust Otsu threshold by: {self.thresh_offset}")

        if "bw_image" in config.attributes.fields:
            self.bw_image = config.attributes.fields["bw_image"].bool_value
        else:
            self.bw_image = False
        self.logger.info(f"Returning black and white image: {self.bw_image}")

        if "mark_detections" in config.attributes.fields:
            self.mark_detections = config.attributes.fields[
                "mark_detections"
            ].bool_value
        else:
            self.mark_detections = False
        self.logger.info(f"Mark the detections on the image: {self.mark_detections}")

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
        # if data manager return result from queue if queue is not empty
        if from_dm_from_extra(extra):
            if not self.capture_all_queue.empty():
                return self.capture_all_queue.get()
            else:
                raise NoCaptureToStoreError()
        try:
            camera = self.dependencies[CameraClient.get_resource_name(camera_name)]
        except KeyError:
            raise ViamError(
                f"Requested camera {camera_name} is not listed in dependencies"
            )
        if isinstance(camera, CameraClient):
            image = await camera.get_image()
            pil_image = viam_to_pil_image(image)
            pil_image, detections = analyze_image(self, pil_image)
            viam_image = pil_to_viam_image(pil_image, image.mime_type)
        else:
            raise ViamError(
                f"Requested camera {camera_name} is not a valid CameraClient"
            )
        result = CaptureAllResult(image=viam_image, detections=detections, extra={})
        id = str(uuid.uuid1())
        self.capture_all_cache.append({id: result})
        result.extra["id"] = id
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
        if isinstance(camera, CameraClient):
            image = await camera.get_image()
            detections = await self.get_detections(image)
        else:
            raise ViamError(
                f"Requested camera {camera_name} is not a valid CameraClient"
            )
        return detections

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        pil_image = viam_to_pil_image(image)
        _, detections = analyze_image(self, pil_image)
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
        if command["command"] == "save_result":
            self.logger.info("save result: %s", command)
            if "result_id" not in command:
                raise ViamError("Missing `result_id` in command")
            else:
                result_id = command["result_id"]
                for item in self.capture_all_cache:
                    if result_id in item:
                        self.capture_all_queue.put(item.get(result_id))
                        return {
                            "result": f"Result with id {result_id} marked for upload"
                        }
                self.logger.error(f"Result with id {result_id} not found in cache")
                raise ViamError(f"Result with id {result_id} not found in cache")
        raise ViamError(f"Unknown command {command}")

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


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
