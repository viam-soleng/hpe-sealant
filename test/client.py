import asyncio

import cv2
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient
from viam.media.utils.pil import viam_to_pil_image
import numpy as np
from typing import Any, Dict

import os
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.getenv("API_KEY"),
        api_key_id=os.getenv("API_KEY_ID"),
    )
    return await RobotClient.at_address(os.getenv("ADDRESS"), opts)


def dict_to_contour(array_dict: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(array_dict["dtype"])
    shape = tuple(int(dim) for dim in array_dict["shape"])
    data = np.array(array_dict["data"], dtype=dtype).reshape(shape)
    return data


async def main():
    machine = await connect()

    # Call the contour detection vision service "vision-sealant" on the robot
    vision_sealant = VisionClient.from_robot(machine, "vision-sealant")
    result = await vision_sealant.capture_all_from_camera(
        "sealant-broken", return_image=True
    )
    # Extract opencv contours from the result
    contours = [dict_to_contour(contour) for contour in result.extra["contours"]]
    print(f"Number of contours detected: {len(contours)}")
    image = viam_to_pil_image(result.image)
    np_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", np.array(np_image))
    # Display raw image for 5 seconds
    # cv2.waitKey(5000)
    # cv2.drawContours(np_image, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("image", np.array(np_image))
    # Display image with contours for 5 seconds
    # cv2.waitKey(5000)
    # Don't forget to close the machine when you're done!

    print(f"Hausdorff Distances: {result.extra["hausdorff"]}")
    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
