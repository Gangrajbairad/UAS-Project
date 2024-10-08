import cv2
import numpy as np

# Function to classify burnt and green grass areas
def segment_grass(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define ranges for burnt grass (brown) and green grass (green)
    burnt_grass_lower = np.array([10, 100, 20])
    burnt_grass_upper = np.array([30, 255, 200])
    
    green_grass_lower = np.array([40, 50, 50])
    green_grass_upper = np.array([90, 255, 255])
    
    # Create masks for burnt grass and green grass
    burnt_mask = cv2.inRange(hsv, burnt_grass_lower, burnt_grass_upper)
    green_mask = cv2.inRange(hsv, green_grass_lower, green_grass_upper)
    
    # Overlay colors to distinguish burnt and green grass
    result = image.copy()
    result[burnt_mask > 0] = (0, 0, 255)  # Red for burnt grass
    result[green_mask > 0] = (0, 255, 0)  # Green for green grass

    return result, burnt_mask, green_mask

# Function to detect houses (red and blue triangles) and calculate priority
def detect_houses(image, burnt_mask, green_mask):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to detect houses
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours for houses
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Variables to store counts and priorities
    houses_on_burnt = 0
    houses_on_green = 0
    priority_burnt = 0
    priority_green = 0
    
    for cnt in contours:
        # Calculate area of the contour (to filter out noise)
        area = cv2.contourArea(cnt)
        if area > 100:  # Threshold for size of houses
            # Get the center of the house to check if it's on burnt or green grass
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Check if the house is on burnt or green grass
                if burnt_mask[cy, cx] > 0:
                    houses_on_burnt += 1
                    priority_burnt += 2 if is_blue(image[cx, cy]) else 1
                elif green_mask[cy, cx] > 0:
                    houses_on_green += 1
                    priority_green += 2 if is_blue(image[cx, cy]) else 1
    
    return houses_on_burnt, houses_on_green, priority_burnt, priority_green

# Helper function to check if a house is blue
def is_blue(pixel):
    return pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 100  # Blue in BGR

# Function to calculate rescue ratio
def calculate_rescue_ratio(priority_burnt, priority_green):
    return priority_burnt / priority_green if priority_green != 0 else float('inf')

# Main function to process images
def process_image(image_path):
    image = cv2.imread(image_path)
    
    # Segment the grass areas
    segmented_image, burnt_mask, green_mask = segment_grass(image)
    
    # Detect houses and calculate priority
    houses_on_burnt, houses_on_green, priority_burnt, priority_green = detect_houses(image, burnt_mask, green_mask)
    
    # Calculate rescue ratio
    rescue_ratio = calculate_rescue_ratio(priority_burnt, priority_green)
    
    # Display results
    print(f"Houses on burnt grass: {houses_on_burnt}, Houses on green grass: {houses_on_green}")
    print(f"Priority on burnt grass: {priority_burnt}, Priority on green grass: {priority_green}")
    print(f"Rescue Ratio: {rescue_ratio}")
    
    # Show the segmented image
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '3'
process_image(image_path)
