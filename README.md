# Close-Range Photogrammetry
A C++ project which implements Resection, Intersection and Direct Linear Transformation in close-range photogrammetric control field of Wuhan University.
## Part 1: Resection+Intersection
The processing of spatial resection is encapsulated within the "CResection" class. In the "Re_Intersection" function, the "cal_resection" 
function is invoked to complete the resection process. Additionally, within the "Re_Intersection" function, spatial forward intersection is 
achieved through point projection and the rigorous solution of collinearity equations. Finally, by calling the "Re_Intersection" function 
within the main function, the computation process is effectively completed.

The resulting accuracy is high, with swift computational speed. The program execution is straightforward, facilitating efficient camera 
calibration. The three-dimensional spatial coordinates obtained from forward intersection not only effectively verify the accuracy of 
the resection model but also provide reference values for the computed three-dimensional spatial coordinates of undetermined points in 
the latter part of the direct linear transformation model.

## Part 2: Direct Linear Transformation(DLT)
The direct linear transformation (DLT) is encapsulated within the "CDLT" class. The DLT algorithm is roughly divided into two parts 
resembling resection and forward intersection, and it is eventually implemented.

The first part is realized through the "GetElement" function, which obtains the "l" coefficients, distortion coefficients, and interior 
and exterior orientation elements, followed by precision assessment.

The latter part is achieved through the "init_XYZ" and "cal_XYZ" functions, which determine the object space coordinates of both reference 
and unknown points. Additionally, the distance between the unknown point in the active control frame and point 52 is calculated. Finally, 
precision assessment is conducted for the reference points.

All these steps can be executed by calling the "DLT" function within the main function.
