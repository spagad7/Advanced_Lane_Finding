# Advanced Lane Finding

The goal of this project is to develop a program to process a video stream from
forward-facing dash-cam video, and output an annotated video stream which
identifies -

* Current lane the vehicle is in
* Radius of curvature of the lane
* Position of the vehicle with respect to the center of the lane

The pipeline to process the input video stream includes following steps -

* Compute camera calibration matrix and distortion coefficients from given set
of chessboard images.
* Undistort each frame/image in input video.
* Apply perspective transform to get top-down("birds-eye-view") of the road
* Improve contrast of the image using adaptive histogram equalization.
* Use color transforms, gradients to create thresholded binary image.
* Use sliding window and neighborhood search algorithms to identify the lane boundaries.
* Determine curvature of the lane and position of the vehicle with respect to
the center of the lane.
* Undo perspective transform to draw the detected lane on the input video.
* Display output image with annotated lane, radius of curvature of lane and
position of the vehicle in the lane.


## Usage
To run the program with calib file

```
python main.py -c=<calib file name> -v=<path to input video file>

Ex: python main.py -c=calib.p -v=videos/project_video.mp4
```

To run the program without calib file

```
python main.py -p=<path to chessboard images> -nx=<number of corners along x axis> -ny=<number of corners along y axis> -c=<calib file name> -v=<path to input video file>

Ex: python main.py -p=camera_cal -nx=9 -ny=6 -c=calib.p -v=videos/project_video.mp4
```


[//]: # (Image References)

[image1]: output_images/chess.png "Distorted Chessboard Image"
[image2]: output_images/chess_undst.png "Undistorted Chessboard Image"
[image3]: output_images/img_dst.png "Sample Distorted Image"
[image4]: output_images/img_undst.png "Sample Undistorted Image"
[image5]: output_images/img_noclahe.png "Image without CLAHE"
[image6]: output_images/img_clahe.png "Image with CLAHE"
[image7]: output_images/img_lab.png "B channel in LAB"
[image8]: output_images/img_hls.png "L channel in HLS"
[image9]: output_images/img_thresh.png "Combined threshold image"
[image10]: output_images/img_orig.png "Input Image"
[image11]: output_images/img_warp.png "Warped Image"
[image12]: output_images/img_lines_rect.png "Sliding Window"
[image13]: output_images/img_lines_norect.png "Neighborhood Search"
[image14]: output_images/formula1.png "Formula 1"
[image15]: output_images/formula2.png "Formula 3"
[image16]: output_images/formula3.png "Formula 4"
[image17]: output_images/formula4.png "Formula 5"
[image18]: output_images/img_output.png "Output Image"


## Files and Functions

### camera.py
I have grouped all the camera related functions, namely, `calibrate`, `saveCalibData`, `loadCalibData`, `undistortImg`, `calcPerspectiveTransform`, `warpImage`, `unwarpImage`, `gradientSobel`, `gradientMag`, `gradientDir`, `applyColorThreshold` and `convertToBinary` in `Camera` class. The description of each of these function can be found below.

* `calibrate` - is implemented in lines 23 through 43 in `camera.py`, and it takes path to chessboard images and number of corners along x and y direction as input and calculates camera calibration matrix and distortion coefficients.
* `saveCalibData` - is implemented in lines 46 through 54 in `camera.py`, and it takes filename as input. It saves camera calibration matrix and distortion coefficients calculated by `calibrate` function in a pickle file.
* `loadCalibData` - is implemented in lines 57 through 62 in `camera.py`, and it takes filename as input. It loads camera calibration matrix and distortion coefficients from a pickle file.
* `undistortImg` - is implemented in lines 65 through 67 in `camera.py`, and it takes distorted image as input and applies distortion correction using the camera calibration matrix and distortion coefficients and outputs undistorted image.
* `calcPerspectiveTransform` - is implemented in lines 70 through 72 in `camera.py`, and it calculates the perspective transform and inverse perspective transform matrices which will be used to warp and unwarp images.
* `warpImage` - is implemented in lines 75 through 84 in `camera.py`, and it takes undistorted image as input and applies perspective transform to the image to get top down view.
* `unwarpImage` - is implemented in lines 87 through 95 in `camera.py`, and it takes warped image as input and undoes the perspective transform to get back the original view.
* `gradientSobel` - is implemented in lines 98 through 109 in `camera.py`, and it takes warped image, orientation, kernel_size and threshold as input, calculates gradient along the specified orientation and outputs thresholded image.
* `gradientMag` - is implemented in lines 112 through 124 in `camera.py`, and it takes warped image, kernel_size and threshold as input, calculates gradient magnitude and outputs thresholded image.
* `gradientDir` - is implemented in lines 127 through 137 in `camera.py`, and it takes warped image, kernel_size and threshold as input, calculates gradient direction and outputs thresholded image.
* `applyColorThreshold` - is implemented in lines 142 through 156 in `camera.py`, and it takes warped image, and color settings as input, converts the image to specified color space and applies threshold
* `convertToBinary` - is implemented in lines 160 through 167 in `camera.py`, and it takes warped image as input, applies combines output of various color and gradient transforms to get a final thresholded image.


### line.py
I have grouped all the line related functions, namely, `findLine`, `lineDetected`, `lineValid`, `findRadCurv` and `drawLine` functions in `Line` class. The description of each of these methods is listed below -  

* `findLine` is implemented in lines 32 through 161 in `line.py`, and it takes warped binary image and side (left/right) as input and applies sliding window and neighborhood search algorithms to identify pixels corresponding to lane boundary and fits a curved line using `polyfit` function in numpy. An instance of Line class is responsible for finding either left or right lane boundaries, not both.
* `lineDetected` is implemented in lines 164 through 175 in `line.py`, and it checks if the unique y coordinates detected by `findLine` are more than a certain threshold or not. This is to ensure that the nonzero pixels in the binary image correspond to a line and not some noise.
* `lineValid` is implemented in lines 179 through 193 in `line.py`, and it checks the validity of a detected line. It verifies that the radius of curvature of the line and average x-coordinate of the line fall within a certain margin of average radius of curvature and average x coordinate of last 10 lines, respectively.
* `findRadCurv` is implemented in lines 197 through 204 in `line.py`, and it calculates the radius of curvature of the line.
* `drawLine` is implemented in lines 208 through 230 in `line.py`, and it draws sliding window rectangles using `rectangle` function in OpenCV, and valid detected lines using `polyines` function in OpenCV.


### main.py
This file contains the code for pipeline and also has a function to draw lanes.

---

## Rubric Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In this step, I generate "object points", which represent (x, y, z) coordinates of all the internal corners of chessboard pattern in the world and store them in `obj_pts`. Next, for every chessboard image, I use `findChessboardCorners` function in OpenCV library to detect all the internal corners of chessboard pattern and use `calibrateCamera` function to get camera calibration matrix and distortion coefficients. The camera is calibrated only for the first time when the program is run and the calibration matrix and distortion coefficients are stored in `calib.p` using `saveCalibData` function. If the `calib.p` file exists then the program reads the calibration matrix and distortion coefficients from the file using `loadCalibData` function, this reduces the initial load time of the program.

After the camera calibration matrix and distortion coefficients are calculated, I use `undistort` function to apply distortion correction to input image. The code for this can be found in lines 23 through 62 in the file `camera.py`.

![alt text][image1]
Fig1: Distorted chessboard image

![alt text][image2]
Fig2: Undistorted chessboard image

### Pipeline
#### 1. Provide an example of a distortion-corrected image.

The first step in the pipeline is to undistort input image and a sample distorted and undistorted image is shown below.

![alt text][image3]
Fig3: Distorted input image

![alt text][image4]
Fig4: Undistorted input image

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I ran many tests to identify the best colorspace, gradients and thresholds to identify yellow and white lines. I found that using gradient method introduced lot of noise (especially in the dark and light sections of the road in challenge video). And I found that the B channel in the range [160, 255] in LAB colorspace is best for detecting yellow lines and L channel in the range [210, 255] in HLS color space is best for detecting white lines.

In addition to finding right color space, I found that the thresholds for B channel in LAB colorspace and L channel in HLS colorpsace, which produced optimal results in the _project-video.mp4_, were too high for detecting yellow and white lines in _challenge-video.mp4_. After further investigation, I found that the contrast of the _challenge-video.mp4_ was quite lower than the _project-video.mp4_. To resolve this problem, I normalized the contrast of the image using **Contrast Limiting Adaptive Histogram Equalization (CLAHE)** algorithm in OpenCV. This improved the image contrast and allowed me to use a common threshold for any type of input image.

![alt text][image5]
Fig5: Image without CLAHE

![alt text][image6]
Fig6: Image with CLAHE

I combined the thresholded image from both colorspaces to get a binary image with both yellow and white lines highlighted. The code for this can be found from lines 142 through 167 in `camera.py`.

![alt text][image7]
Fig7: B channel in LAB colorspace

![alt text][image8]
Fig8: L channel in HLS colorspace

![alt text][image9]
Fig9: Combined thresholded image


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code to perform perspective transform includes a function called `calcPerspectiveTransform` which uses the source points in the undistorted image and destination points in top-down ("birds-eye") view to calculate transform and inverse transform matrices.

| Source        | Destination   |
|:-------------:|:-------------:|
| 530, 470      | 10, 10        |
| 715, 470      | 190, 10       |
| 990, 645      | 190, 290      |
| 170, 645      | 10, 290       |

These values are hardcoded in the program and can be found in lines 80 through 90 in `main.py`. And the size of destination image is 250 x 300. I choose small values for destination points and destination image size because it's faster to convert small image to binary and find lane lines.

![alt text][image10]
Fig10: Input image

![alt text][image11]
Fig11: Warped image

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The algorithm implemented in `findLine` can be summarized as follows:

1. Find histogram of lower half of the image and identify peaks in the left and right half of the image.
2. Get indices of nonzero pixels in the binary image.
3. **Sliding Window Algorithm:** If line not detected in previous frame, for each window in windows list, get indices of pixels which lie inside the window. If the number of pixels found in the window is greater than a threshold then update the peak value (found initially using histogram of lower half of the image) with the average of x coordinates of pixels lying inside the window.
4. **Neighborhood Search Algorithm:** If line detected in the previous frame, then search in the neighborhood of the line found in previous frame. This improves the efficiency of algorithm as the search space for finding the new line is reduced.
5. Check if the pixels detected by the two algorithms is actually a line using `lineDetected` function.
6. If `lineDetected` returns true, check for validity of the line using `lineValid` function. If the detected line is valid then update the line measurements for the `drawLine` function to draw.
7. If `lineDetect` or `lineValid` check fails, then calculate a new line using average of last 10 line measurements.

![alt text][image12]
Fig12: Detected lines using Sliding Window

![alt text][image13]
Fig13: Detected lines using Neighborhood Search


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate radius of curvature, I used the formula from [this](http://mathworld.wolfram.com/RadiusofCurvature.html) website.

![alt text][image14]

where,

![alt text][image15]

![alt text][image16]

so,

![alt text][image17]

The implementation of this formula can be found in lines 197 through 204 in `line.py`. To find the position of the vehicle with respect to the center of the lane, I calculated the difference between the midpoint along x axis of the two detected lane lines and center of the warped image. The implementation of this can be found in lines 141 through 143 in `main.py`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the function `drawLane` in lines 13 through 37 in `main.py`, which takes the undistorted image and two instance of Line class, one corresponding to left boundary of the lane and the other to the right boundary. This function gets the measurements of the line calculated by function in Line class and draws a lane using `fillPoly` function in OpenCV. Finally, the output pipeline video is implemented in `main` function in lines 118 116 through 161 in `main.py`

![alt text][image18]
Fig14: Output image

---

### Pipeline (video)

**Note:** In output1 and output2 you don't see any search window rectangles because the Neighborhood Search algorithm is finding lines in the neighborhood of previously found line. To see the Sliding Window algorithm in action, please check second video.

* Link to output on _project_video.mp4_ with Sliding Window and Neighborhood Search [output1](videos/output1_norect.mp4)
* Link to output on _challenge_video.mp4_ with Sliding Window and Neighborhood Search [output2](videos/output2_norect.mp4)
* Link to output on _project_video.mp4_ with only Sliding Window [output3](videos/output1_rect.mp4)
* Link to output on _challenge_video.mp4_ with only Sliding Window [output4](videos/output2_rect.mp4)

---

### Discussion

#### Challenges with thresholding
I spent more than 50% of the project time on thresholding as it is the basis for lane detection. Below are some of the challenges I faced and how I overcame them.

One of the hardest part of the project was to find the right colorspace, gradient method and threshold which works with all the project-videos. I was not keen on altering thresholds for different videos. To overcome this challenge, I wrote the script `tests\getColor.py` to get value of pixel in different color spaces with mouse clicks. This script is not so elegant, I just wrote it as a makeshift tool to help me identify correct thresholds. Using this tool I noted down the range of pixel values of yellow and white lines in different color spaces and in different images. This helped me to get correct threshold values to retain only yellow and white lines in thresholded image.

Next, I wrote the script `tests\threshTest.py` to try various combinations of multiple color spaces, gradients and thresholds. Using this script, I found that the B channel in LAB colorspace to be the best one (better than S channel in HLS) for identifying yellow lines, and L channel in HLS to be the best for identifying white lines.

Finally, as discussed in the second pipeline rubric point, I used CLAHE to enhance the contrast of the image. This enabled me to use common threshold values for both _project-video.mp4_ and _challenge-video.mp4_.

#### Challenges with shadows and wobbly lines
The lane detection system implemented in this project loses track of the lanes when there is large patch of shadow, this can be especially seen in the section of _challenge-video.mp4_ where the car goes under a bridge. To overcome this challenge, I maintained a list of measurements of previous 10 lines, and when the lane is not detected or when the detected lane is invalid, the algorithm uses average of previous 10 lines to predict the line. This is method works well in _project-video.mp4_ and _challenge_video.mp4_, but it fails in the _harder-challenge-video.mp4_ as the road is very curvy and lots of dull and overexposed regions.

#### Challenges with program running speed
Initially, I made a mistake of calculating perspective and inverse perspective transform for every frame in the video, the video output was quite slow and jittery. Calculating inverse of a matrix is an expensive process, to get over this problem, I pre-calculated the perspective transform and inverse perspective transform before running the frames through the pipeline.

The performance of the program also depends on the way different steps are ordered in the pipeline. One of the example of bad pipeline is

1. Undistort Image
2. Apply threshold to undistorted image and convert the image to binary image
3. Warp image
4. Detect Lines ...

I improved the performance of my program by reordering the steps in the pipeline to

1. Undistort Image
2. Warp image
3. Apply threshold to warped image and convert the image to binary image
4. Detect Lines ...

My warped image is quite small (250 x 300), which enables faster detection of lines and more realtime performance.

#### Areas of improvement
There is still a lot of scope of improvement in this project, below are some of them which I can think of,

* Rather than using average of previous n measurements, Kalman filter can be used to make lane detection system more robust.
* Currently, the program doesn't give good results on _harder-challenge-video.mp4_. More experiments can be run to detect better combination of colorspaces, gradients and threshold.
