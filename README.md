# Module OpenCV Contours Detection and Comparison

This module uses OpenCV contour features to detect such, filter and compare them. For example you could use this module to store reference contours and then use other images to find the same contours and compare them to the stored references.

## Model "hpe-automotive:sealant-check:sealant"

This module implements a Viam `VisionService`.

### Configuration

#### Attributes

The following attributes are available for this model:

| Name            | Type   | Inclusion | Description                                                                                           |
| --------------- | ------ | --------- | ----------------------------------------------------------------------------------------------------- |
| `draw_contours` | string | Optional  | Draws the available contours on the returned image ["reference", "detected", "both"]. Default is none |
| `max_contours`  | int    | Optional  | Maximum number of contours processed. Default or `0` is unlimited.                                    |
| `min_area`      | float  | Optional  | Minimal contour area to be considered                                                                 |
| `max_area`      | float  | Optional  | Maximal contour area to be considered                                                                 |
| `min_height`    | int    | Optional  | Minimal contour height to be considered                                                               |
| `max_height`    | int    | Optional  | Maximal contour height to be considered                                                               |
| `min_width`     | int    | Optional  | Minimal contour width to be considered                                                                |
| `max_width`     | int    | Optional  | Maximal contour width to be considered                                                                |
| `bw_image`      | int    | Optional  | Return the thresholded image                                                                          |
| `thresh_offset` | int    | Optional  | Adjust the Otsu threshold [internal formula: threshold = otsu-threshold - thresh_offset]              |

#### Example Configuration

```json
{
  "max_area": 3127349,
  "max_width": 2000,
  "draw_contours": "both",
  "max_contours": 10,
  "min_area": 1127349
}
```

#### Configuration Assistance

If you configure debug logs for the vision service component (restart required), you will see the detected contours width, height, area and arclength in the logs!

### DoCommand

This module provides two `do_commands`, one for storing rerference contours and the second one to delete the stored reference contours.

#### Save Results

```json
{
  "command": "save_result",
  "result_id": "xxxx"
}
```

## Publish New Release

The repository contains a git action which triggers on the following command:

```
git tag x.x.x
git push origin --tag x.x.x
```
