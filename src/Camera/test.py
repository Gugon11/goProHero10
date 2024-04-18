import cv2
import numpy as np

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

        # Extract centroids and draw them on the result image
        centroids = []
        for cnt in filtered_contours:
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                centroids.append((cX, cY))
                cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return centroids, result

    def detect_red_cars(image):
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
        centroids = []
        for cnt in filtered_contours:
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                centroids.append((cX, cY))
                cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return centroids, result

    def detect_pink_cars(image):
        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for pink color in HSV
        lower_pink1 = np.array([130, 50, 50])
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

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        min_area = 500
        max_area = 2500
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # Draw the filtered contours on the original image
        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        # Extract centroids and draw them on the result image
        centroids = []
        for cnt in filtered_contours:
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                centroids.append((cX, cY))
                cv2.circle(result, (cX, cY), 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return centroids, result
    
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
    all_centroids = blue_centroids + red_centroids + pink_centroids

    return all_centroids, combined_result

image = cv2.imread("goProHero10/src/Camera/images/linear/img_0000.png")

centroids, res = detect_cars(image)
print(centroids)
im = cv2.resize(res, (960, 540))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
