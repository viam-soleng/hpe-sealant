# Current Work in Progress

1. Check Contours Distance

- https://answers.opencv.org/question/204390/using-opencv-distance-transform-to-find-width-of-a-curve/
- Hausdorff is perfectly suited: https://en.wikipedia.org/wiki/Hausdorff_distance

## Description

### Reference Shapes for Filtering

1. Detect Sealant Shape references (2 contours)
2. Store reference shapes in pickle file

### Verify Sealant Line

1. Detect contours
2. Filter based upon the stored references -> need two for calculation
3. Use Hausdorff to calculate min and max distance of the two contours
4. Convert distance into `mm`

### Display Result

1. Compare distance to threshold set in `mm`
2. Update display accordingly
