import cv2
import numpy as np
import pandas as pd


#-------------This test.py serves as an aux to test some functions--------------------------------

def detect_cars(image):
    def detect_blue_cars(image):
        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for blue color in HSV
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([120, 255, 255])

        # Create a mask using the specified range
        mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        min_area = 500
        max_area = 1500
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # Draw the filtered contours on the original image
        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        # Extract the centroid of the largest contour
        centroid = None
        max_contour_area = 0
        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            if area > max_contour_area:
                moments = cv2.moments(cnt)
                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])
                    if 600 <= cX <= 1650 and cY > 85:  # Check if centroid is within the specified bounds
                        centroid = (cX, cY)
                        max_contour_area = area
        
        # Draw the centroid on the result image
        if centroid is not None:
            cv2.circle(result, centroid, 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            
        return centroid, result

    def detect_red_cars(image):
        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        '''# Define the lower and upper bounds for red color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 100, 100])
        upper_red2 = np.array([180, 255, 255])'''
        lower_yellow = np.array([20, 100, 100])   # Lower bound for yellow
        upper_yellow = np.array([30, 255, 255])   # Upper bound for yellow


        # Create masks using the specified ranges
        mask1 = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        #mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)


        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        min_area = 500
        max_area = 1500
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # Draw the filtered contours on the original image
        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        # Extract the centroid of the largest contour
        centroid = None
        max_contour_area = 0
        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            if area > max_contour_area:
                moments = cv2.moments(cnt)
                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])
                    if 600 <= cX <= 1650 and cY > 85:  # Check if centroid is within the specified bounds
                        centroid = (cX, cY)
                        max_contour_area = area
        
        # Draw the centroid on the result image
        if centroid is not None:
            cv2.circle(result, centroid, 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return centroid, result

    def detect_pink_cars(image):
        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for pink color in HSV
        lower_pink1 = np.array([150, 50, 50])
        upper_pink1 = np.array([170, 255, 255])

        lower_pink2 = np.array([0, 50, 50])
        upper_pink2 = np.array([10, 255, 255])

        # Create masks using the specified ranges
        mask1 = cv2.inRange(image_hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(image_hsv, lower_pink2, upper_pink2)

        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        min_area = 500
        max_area = 2500
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # Draw the filtered contours on the original image
        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        # Extract the centroid of the largest contour
        centroid = None
        max_contour_area = 0
        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            if area > max_contour_area:
                moments = cv2.moments(cnt)
                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])
                    if 600 <= cX <= 1650 and 85 <= cY <= 990:  # Check if centroid is within the specified bounds
                        centroid = (cX, cY)
                        max_contour_area = area
        
        # Draw the centroid on the result image
        if centroid is not None:
            cv2.circle(result, centroid, 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return centroid, result
    
    
    def adjust_intensity(image, alpha=0.35, beta=0):
        # Perform intensity adjustment using alpha blending
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

    # Detect blue cars
    blue_centroids, blue_result = detect_blue_cars(image)

    # Adjust intensity of blue result
    blue_result_adjusted = adjust_intensity(blue_result)

    # Detect red cars
    red_centroids, red_result = detect_red_cars(image)

    # Adjust intensity of red result
    red_result_adjusted = adjust_intensity(red_result)

    # Detect pink cars
    pink_centroids, pink_result = detect_pink_cars(image)

    # Adjust intensity of pink result
    pink_result_adjusted = adjust_intensity(pink_result)

    # Combine adjusted results
    combined_result = cv2.add(cv2.add(blue_result_adjusted, red_result_adjusted), pink_result_adjusted)

    # Combine centroids
    all_centroids = [blue_centroids, red_centroids, pink_centroids]

    return all_centroids, combined_result

def crop_img(image, x, y, h, w):
    cropped_img = image[y:y+h, x:x+w]
    return cropped_img

def show_image(title, img):
    # Resize image to fit screen for better visualization
    im_resized = cv2.resize(img, (960, 540))
    cv2.imshow(title, im_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_circle(img):
    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #show_image("Gray image", gray)

    # Adaptive thresholding for better binary conversion
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #show_image("Binary Image", img_bin)

    # Opening binary image to remove noise around the circles
    close_kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_bin, close_kernel, iterations=1)
    img_closed = cv2.erode(img_dilated, close_kernel, iterations=1)
    #show_image("Opened Image", img_closed)

    # Find circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        img_closed, 
        cv2.HOUGH_GRADIENT, 
        dp=1.0, 
        minDist=50, # Adjusted minimum distance between circles
        param1=50, 
        param2=25, 
        minRadius=10, # Set a reasonable minimum radius
        maxRadius=30 # Set a reasonable maximum radius
    )

    # List to store the centers of the circles
    centers = []

    # If circles are detected, draw them on the original image and store the centers
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            centers.append(center)
            # Draw the outer circle
            cv2.circle(img, center, i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            #cv2.circle(img, center, 2, (0, 0, 255), 3)
    
    return img, centers

image = cv2.imread("goProHero10/src/Camera/images/linear/img_checkpoints.png")
#image = crop_img(image, 600, 80, 1000, 1200)

circles, center= find_circle(image)
centroids, circles = detect_cars(image)

center = np.array(center)
#centroids = np.array(centroids)

cv2.circle(circles, center[0], 2, (0, 0, 255), 3) #red
cv2.circle(circles, center[1], 2, (255, 0, 0), 3) #blue
cv2.circle(circles, center[2], 2, (0, 255, 0), 3) #green
cv2.circle(circles, center[3], 2, (0, 0, 0), 3)   #black
cv2.circle(circles, center[4], 2, (255, 255, 255), 3) #white
cv2.circle(circles, center[5], 2, (0, 222, 255), 3) #yellow
cv2.circle(circles, center[6], 2, (230, 0, 255), 3) #pink
cv2.circle(circles, center[7], 2, (230, 255, 0), 3) #baby blue
cv2.circle(circles, center[8], 2, (0, 111, 255), 3) #orange


#Difference between car coordinates and Origin coordinates
#centroids_origin = centroids -  center

#Ratio of pixel to millimeter obtained in pixel2mm.py
px2mm = 0.6337807227544455

#center_cm = (center/px2mm)/10 #centimeter
#centroids_cm = (centroids_origin/px2mm)/10 #centimeter

'''print("Blue car: ", centroids_cm[0])
print("Yellow car: ", centroids_cm[1])
print("Pink car: ", centroids_cm[2])'''
circles_organized = np.array([center[6],
                              center[4],
                              center[5],
                              center[2],
                              center[0],
                              center[3],
                              center[8],
                              center[7],
                              center[1]])

print("Checkpoints:", circles_organized)

im = cv2.resize(circles, (960, 540))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

