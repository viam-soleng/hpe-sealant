# hpe-sealant

This module wraps opencv contour finding functions into a Viam module. It returns the contours found in an image as part of the `extra` attribute when calling `capture_all_from_camera()`. Additionally it returns the bounding boxes of the contours as detections.

This module also contains a sample client script, explaining how to work with the returned data.

To get this module working, you must add a camera as a depedency in the configuration!


The module allows storing reference contours using the following do_command():

```json
{
  "command":"save_contours",
  "camera_name":"camera_name"
}
```
You can also delete the stored contours with the following command:

```json
{
  "command":"delete_contours"
}
```

