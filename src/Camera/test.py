import cv2
import numpy as np
#----------------------------------------------------------
def detect_blue_car(image):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in HSV
    lower_blue = np.array([90, 150, 150])
    upper_blue = np.array([120, 255, 255])

    # Create a mask using the specified range
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    max_area = 1400
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    # Create an empty mask for the filtered contours
    filtered_mask = np.zeros_like(mask)

    # Draw the filtered contours on the mask
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # Apply the filtered mask to the original image
    result = cv2.bitwise_and(image, image, mask=filtered_mask)

    # Extract centroids
    moments = cv2.moments(filtered_mask)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    # Draw centroid on the result image
    cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
    cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result

#-----------------------------------------------------------------------------------------------------------
def detect_red_car(image):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0,140,140])
    upper_red = np.array([7,255,255])

    # Create a mask using the specified range
    mask = cv2.inRange(image_hsv, lower_red, upper_red)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    max_area = 1400
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    # Create an empty mask for the filtered contours
    filtered_mask = np.zeros_like(mask)

    # Draw the filtered contours on the mask
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # Apply the filtered mask to the original image
    result = cv2.bitwise_and(image, image, mask=filtered_mask)

    # Extract centroids
    moments = cv2.moments(filtered_mask)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    # Draw centroid on the result image
    cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
    cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result
#----------------------------------------------------------------------------------------------------------



# Example usage:
image = cv2.imread("goProHero10/src/Camera/images/linear/img_0000.png")
result_with_car = detect_red_car(image)

im = cv2.resize(result_with_car, (960, 540))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
