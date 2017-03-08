
## **End-to-end Driving Behavioral Cloning Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original1 Image"
[image2]: ./camera_cal/undistorted.jpg "Undistorted Image"
[image3]: ./test_images/test5.jpg "Original2 Image"
[image4]: ./output_images/test5.jpg_color_binary.png "Color Binary Image"
[image5]: ./output_images/straight_line1_warped_binary.png "Binary Warped"
[image6]: ./output_images/histogram_output.png "Histogram"
[image7]: ./output_images/search.png "Search"
[image8]: ./output_images/result.png "TestImResult"
[image9]: ./output_images/test5.jpg_combined_binary.png "TestImResult" 

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---
###Camera Calibration

First step of a computer vision problem where approximate calculation of some dimensions are needed is camera calibration. Camera calibration is calcution of intrinstic parameters like optical center of camera or focal length in different axis and determination of optical distortion coefficients like tangential or radial or both due usage of lens. In [OpenCV documentation page](http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html?) great explanation of these calculation may find further.

Here in this project, pre-captured 9x6 chessboard pattern has been used for parameter and coefficient calculations. First object points are prepared as same as chessboard pattern and then chessboard corners in image plane are found using cv2.findChessboardCorners function and append to image points. 

Afterwards, `cv2.calibrateCamera` function has been used with both object points and image points matrices to get distortion coefficient ("dst") estimation and calibration matrix ("mtx") calculations. Output matrices has been saved for further undistortion of images. All of these calculations could be found in "undistortion.ipynb" file.


### Pipeline (single images)

#### 1. Undistortion of images

Undistortion file just called input file read in all process loops and images are rectified with function `cv2.undistort()`. "mtx" and "dst" are used as an input of this OpenCV function.

Raw camera input image and rectified image is below;

| Source Image             | Undistorted Image           | 
|:------------------------:|:---------------------------:| 
| ![Original Image][image1]| ![Undistorted Image][image2]| 


#### 2. Thresholding

To find most image features in recognizable conditions for further value extractions, image edges should be sharpen and all feature edges should robust to environmental conditions like brightness etc. For this purpose different characteristics of lane line has been considered and line curvature, line color and line shape like continues or dashed has been choosed as to be focused features. And also brightness changes caused by a shadow or a bad weather are considered as worst case scenarios. So HLS color domain for its less ilumunation dependent response is used. L channel of HLS color domain is extracted and Sobel edge detection for x gradient is used with magnitude normalization and resulting image is logically and operated with color thresholding. In below; we can see Sobel x-gradient thresholding in green and color thresholding in blue.


| Source Image             | Colored Binary Image        | Binary Image               |
|:------------------------:|:---------------------------:|:--------------------------:| 
| ![original image][image3]| ![color bin image][image4]  |![comb image][image9]       | 


#### 3. Perspective transform, "Bird-eye view"

Next step for this project is to find bird-eyed view to make further calculations easier. A method called warped transform or perspective transform is a way to make it possible within OpenCV by using source and destination matrices. Required input arguments, source and destination matrices, can found with two different approach to get this top-down view. First is to determine vanishing point of a scene and selecting four points in two lines which are intersect each-other on the vanishing point. After this selection process four points give us a trapezoidal shape. In order to implement these calculations vanishing points are determined at where lane lines of straight driving image intersect. Here lane line detections has been made via Hough Transform. And then with known y-axis values of points (0 and 450, choosen from observations) x-axis values are calculated. Second method can be thought manually tuning. 

For this project hybrid method is used. Source matrix calculated by first method and manually fine-tuned. Destination matrix is just a four points of image corners.

After determination of these matrices `cv2.warpPerspective` has been used and below transformations are obtained.

| Source Image             | Warped Image                | 
|:------------------------:|:---------------------------:|
| ![original image][image3]| ![color bin image][image5]  |

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 375, 480      | 0, 0          | 
| 905, 480      | 685, 0        |
| 1811, 685     | 685, 1280     |
| -531, 685     | 0, 1280       |

****
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 4. Lane-line detection and second order polynomial fit

Bird-eyed view lane then later thresholded for now to detect lane-lines. Thresholded images are consist of ones and zeros and they are convinient to count in x-axis or y-axis of histograms. For our problem starting point of lane-lines are detected where histogram plot of thresholded image makes peak on left side and right of image center on x-axis. For details `lane_histogram` and `lane_peaks` functions can be found on pipeline. In below one can found histogram plot of thresholded image and box searched - polynomial fitted lane-line image


![histogram][image6]
![box-search][image7]

By determination of starting point of lane line in a frame, window searching is implemented for efficient determination of full lane-line points. `sliding_window_search` function looks for minimum number of pixel consists a "1" in defined box and it starts from the previous window to look around. By doing this approach curvy or dashed lane points can be appended in a lane-line array for polynomial fit. 

#### 5. Improvements for jittery lane detections

Frame by frame lane line detections could cause non-accurate detections as well. Base coordinate detections from histogram plot peaks may be wrongly detected or sliding windows may follow wrong curvatures that caused by a shadow or patch on the road. To overcome this issues base lane line buffers and curvature buffers are defined. These buffers are searching for verified detections and recording their values into the array. When there is wrongly determined lane line occur, its base line coordinates and curvature used from previous frame. Here criteria for base line coordinates is lane width and for the curvature is %85 correlation with previous curvature.

#### 6. Radius of curvature calculations

I did this in lines # through # in my code in function `radius_of_curvature`

#### 7. Result of test an image

I implemented this step in lines # through # in my code in function`visualizeWeighted` of pipeline.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Result of test video

Here's a [link to my video result](./project_output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here in this project advanced thresholding techniques for better acquisition of image features and for making lane-lines more clear are implemented. Sobel edge detection with x-axis gradients or y-axis gradients are studied. Color thresholding has been used for different color spaces. Perspective transform has been used for to use histogram counting method to easily find starting point of lane-line. Then sliding window search technique is implemented for more efficient 2nd order polynomial curvature on to the lane. And finally by using camera undistortion how much car is deviation during a ride with respect to lane center is written on test video result.

This study requires many more approaches to make it more robust on harder cases. Lane-line smoothing and implementation of sanity checks for wrong determination of lanes in some frames is replaced with a lane-line is already verified at past.
