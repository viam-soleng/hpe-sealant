import argparse
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
    """Converts the dictionary representation of contours to a numpy array."""
    dtype = np.dtype(array_dict["dtype"])
    shape = tuple(int(dim) for dim in array_dict["shape"])
    data = np.array(array_dict["data"], dtype=dtype).reshape(shape)
    return data


async def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process some contours")
    parser.add_argument("--cmd", type=str, help="Command to execute")
    args = parser.parse_args()

    # Connect to Viam machine
    machine = await connect()
    # Get the vision client for the "vision-sealant" service
    vision_sealant = VisionClient.from_robot(machine, "vision-sealant")
    # Depending on command line argument, execute different commands
    if args.cmd == "save":
        result = await vision_sealant.do_command(
            {"command": "save_contours", "camera_name": "sealant-ref"}
        )
        print(result)
    elif args.cmd == "delete":
        result = await vision_sealant.do_command({"command": "delete_contours"})
        print(result)
    elif args.cmd == "compare":
        result = await vision_sealant.capture_all_from_camera(
            "sealant-ref", return_image=True, return_detections=True
        )
        # print(f"# of contours detected: {len(result.detections)}")
        print("\n")
        print(f"Viam Detections:")
        print(f"{result.detections}")
        print("\n")
        for idx, ctr in enumerate(result.extra["contours"]):
            print(f"Contour {idx}:")
            print(f"Area: {ctr['area']}")
            print(f"ArcLength: {ctr['arclength']}")
            print(f"Hausdorff distances: {ctr['hausdorff']}")
            print("\n")
    else:
        print("Choose a valid command: save, delete, compare")
    # Don't forget to close the machine when you're done!
    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
