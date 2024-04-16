import cv2
import numpy as np
#----------------------------------------------------------
def detect_blue_car(image):
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

    # Extract centroids and draw them on the result image
    for cnt in filtered_contours:
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result
#-----------------------------------------------------------------------------------------------------------
def detect_red_car(image):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks using the specified ranges
    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

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

    # Extract centroids and draw them on the result image
    for cnt in filtered_contours:
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result

#----------------------------------------------------------------------------------------------------------
def detect_pink_car(image):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for pink color in HSV
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])

    # Create a mask using the specified range
    mask = cv2.inRange(image_hsv, lower_pink, upper_pink)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    min_area = 500
    max_area = 2500
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Draw the filtered contours on the original image
    result = image.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # Smooth contours before drawing
    for cnt in filtered_contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)

    # Extract centroids and draw them on the result image
    for cnt in filtered_contours:
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result
#----------------------------------------------------------------------------------------------------------
# Example usage:
image = cv2.imread("goProHero10/src/Camera/images/linear/img_0000.png")
result_with_car = detect_pink_car(image)

im = cv2.resize(result_with_car, (960, 540))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
