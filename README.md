# UAS-Project
Rescue Mission Optimization: 
Image Segmentation and Rescue Priorities
1.	Introduction 
This project is part of a Search and Rescue mission, where I am tasked with processing UAV-captured images of a disaster area. The goal is to differentiate between burnt and green grass, detect houses (represented as colored triangles), and calculate a rescue priority ratio based on the houses' location and priority.

2. Objectives
The key objectives of this project are:
1.	Segment the image into burnt grass (affected areas) and green grass (safe areas).
2.	Detect houses based on shape and color (blue/red triangles), classifying them by their location (burnt/green grass).
3.	Calculate house priorities, where red houses are priority 1, and blue houses are priority 2.
4.	Compute the rescue ratio to optimize decision-making in the rescue operation.
5.	Output the results for all images, providing segmented visuals and calculated statistics for rescue prioritization.

3. Procedure
The project follows a step-by-step approach using Python, NumPy, and OpenCV for image segmentation and feature detection. Here is the breakdown:
Step 1: Image Segmentation
•	Goal: Identify and classify areas of the image as burnt grass and green grass.
•	Method: Use OpenCV to convert the image to the HSV color space and create masks to segment areas based on color thresholds.
Step 2: House Detection
•	Goal: Detect houses represented as blue and red triangles in the image.
•	Method: Using contour detection in OpenCV, we identify house-like shapes and classify them based on their color. Blue houses have a priority of 2, and red houses have a priority of 1.
Step 3: Calculate Priorities and Rescue Ratio
•	Goal: Calculate the total priority of houses on burnt and green grass, and compute the rescue ratio.
•	Formula:  Pr=Pb/Pg, where:
o	Pb = Priority of houses on burnt grass
o	Pg = Priority of houses on green grass
Step 4: Output and Visualization
Goal: Provide segmented images showing the burnt and green grass, and generate a report with house counts, priorities, and rescue ratios.

4. Code Explaination
1. Code for segmentation of images
	Convert image  to HSV color space
	hsv = cv2.cvtcolor(image, cv2.COLOR_BGR2HSV)
•	Because OpenCV reads images in BGR (Blue-Green-Red) format by default, but in HSV (Hue-Saturation-Value) the color (hue) is separated from its intensity (value), making it easier to detect certain color ranges.
	Define Color Ranges for Burnt and Green Grass
burnt_grass_lower = np.array([10, 100, 20])
burnt_grass_upper = np.array([30, 255, 200])
green_grass_lower = np.array([40, 50, 50])
green_grass_upper = np.array([90, 255, 255])
•	To identify burnt grass (brown) and green grass, we need to define their HSV ranges. These ranges allow OpenCV to create masks (binary images) where pixels within the range are marked as 1 (white), and others are 0 (black).
•	The np.array function is used to define the lower and upper bounds for each color. For example:
	Create mask for burnt and green grass
burnt_mask = cv2.inRange(hsv, burnt_grass_lower, burnt_grass_upper)
green_mask = cv2.inRange(hsv, green_grass_lower, green_grass_upper)
•	To identify burnt grass (brown) and green grass, we need to define their HSV ranges. These ranges allow OpenCV to create masks (binary images) where pixels within the range are marked as 1 (white), and others are 0 (black).
•	The cv2.inRange() function generates a binary mask for each color range. In the burnt_mask, all pixels falling within the brown HSV range will be white (255), and others will be black (0). Similarly, green_mask will highlight pixels within the green HSV range.
	Overlay colors on burnt and green grass
result = image.copy()
result[burnt_mask > 0] = (0, 0, 255)  # pink for burnt grass
result[green_mask > 0] = (0, 255, 0)  # yellow for green grass
•	A copy of the original image (result = image.copy()) is created so that we can overlay new colors without modifying the original.
•	Pixels where the mask (burnt_mask or green_mask) is greater than 0 are colored


2.	Code for detection and counting of houses
	Convert image to grayscale

gray = cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)
•	Converting the image to grayscale reduces it to a single channel, simplifying contour detection.

	Thresholding for binary image 
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
•	Thresholding is used to create a binary image, where pixels are either 0 or 255. If a pixel value is above 150 (the threshold value), it will be set to 255 (white); otherwise, it will be 0 (black)
•	Binary Inversion (cv2.THRESH_BINARY_INV): This inverts the threshold, making dark areas (houses) white and the background black.

	Contour detection 
contours = cv2.findcontours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
•	Contours are continuous lines or curves that bound or cover the full boundary of a shape. OpenCV’s findContours() function retrieves these boundaries from the binary image. 
•	 cv2.RETR_EXTERNAL ensures only external contours are detected, which are important for identifying house shapes.
•	cv2.CHAIN_APPROX_SIMPLE is used to compress the contours and save memory.

	Calculate moments and determine house position
M = cv2.moments(cnt)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])  # X coordinate of the centroid
    cy = int(M['m01'] / M['m00'])  # Y coordinate of the centroid
•	the centroid (geometric center) of a shape can be calculated using moments (cx, cy), which helps us determine whether the house is on burnt or green grass.
3.Code for calculate priority and rescue ratio 
if burnt_mask[cy, cx] > 0:
    houses_on_burnt += 1
    priority_burnt += 2 if is_blue(image[cx, cy]) else 1
else if green_mask[cy, cx] > 0:
    houses_on_green += 1
    priority_green += 2 if is_blue(image[cx, cy]) else 1
def calculate_rescue_ratio(priority_burnt, priority_green):
    return priority_burnt / priority_green if priority_green != 0 else float('inf')  
•	Once we have the centroid of the house (cx, cy), we check whether it lies on burnt grass or green grass using the masks. If it's on burnt grass, the count and priority are updated accordingly.
•	Priority Assignment: If the house is blue (checked by the helper function is_blue()), its priority is 2; if red, its priority is 1
5.Output
n_houses=[[6,4],[5,4],[8,6],[3,4],[6,3],[6,4],[2,6],[7,5],[6,4],[6,4]]
priority_houses=[[8,6],[8,6],[12,10],[4,7],[9,4],[10,5],[3,10],[10,8],[8,5],[8,6]]
priority_ratio=[1.33,1.33,1.2,0.59,2.25,2,0.3,1.25,1.6,1.33]
image_by_rescue_ratio=[image5,image6, image9, image1, image2, image10, image8, image3, image4, image7]


Resources
 •  OpenCV Documentation
 •  NumPy Documentation
 •  Python Documentation 
 . youtube.com/channel/UCjWY5hREA6FFYrthD0rZNIw
   Python 3.7: https://www.python.org/downloads/release/python-378/ 
●  Python 3: https://automatetheboringstuff.com/ 
●  OpenCV: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html [Video Lecture] 
●  GitHub : https://docs.github.com/en/get-started/quickstart/hello-world  

