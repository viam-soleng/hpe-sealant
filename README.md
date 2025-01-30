# Module OpenCV Contours Detection and Comparison

This module uses OpenCV contour features to detect such, filter and compare them. For example you could use this module to store reference contours and then use other images to find the same contours and compare them to the stored references.

## Model hpe-automotive:sealant-check:sealant"

This module implements a Viam `VisionService`.

### Configuration

The module works without configuration but limiting the detected contours is always recommended:

\`\`\`json
{
"draw_contours": "both",
"max_contours": 10
}
\`\`\`

#### Attributes

The following attributes are available for this model:

| Name            | Type   | Inclusion | Description                                                                                           |
| --------------- | ------ | --------- | ----------------------------------------------------------------------------------------------------- |
| `draw_contours` | string | Optional  | Draws the available contours on the returned image ["reference", "detected", "both"]. Default is none |
| `max_contours`  | int    | Optional  | Maximum number of contours processed                                                                  |
| `min_area`      | float  | Optional  | Minimal contour area to be considered                                                                 |
| `max_area`      | float  | Optional  | Maximal contour area to be considered                                                                 |
| `min_height`    | int    | Optional  | Minimal contour height to be considered                                                               |
| `max_height`    | int    | Optional  | Maximal contour height to be considered                                                               |
| `min_width`     | int    | Optional  | Minimal contour width to be considered                                                                |
| `max_width`     | int    | Optional  | Maximal contour width to be considered                                                                |

#### Example Configuration

\`\`\`json
{
"max_area": 3127349,
"max_width": 2000,
"draw_contours": "both",
"max_contours": 10,
"min_area": 1127349
}
\`\`\`

### DoCommand

TODO: If your model implements DoCommand, provide an example payload of each command that is supported and the arguments that can be used. If your model does not implement DoCommand, remove this section.

#### Example DoCommand

\`\`\`json
{
"command_name": {
"arg1": "foo",
"arg2": 1
}
}
\`\`\`
