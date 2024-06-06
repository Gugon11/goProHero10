import numpy as np
import cv2

def find_circle(img):
    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better binary conversion
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Opening binary image to remove noise around the circles
    close_kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_bin, close_kernel, iterations=1)
    img_closed = cv2.erode(img_dilated, close_kernel, iterations=1)

    # Find circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        img_closed, 
        cv2.HOUGH_GRADIENT, 
        dp=1.0, 
        minDist=100,  # Adjusted minimum distance between circles
        param1=50, 
        param2=25,  # Adjusted parameter for circle detection sensitivity
        minRadius=10,
        maxRadius=30
    )

    # List to store the centers of the circles
    centers = []

    # If circles are detected, draw them on the original image and store the centers
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = np.array([i[0], i[1]])
            centers.append(center.tolist())  # Append as a list
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    circle_center = np.array([centers])

    return img, circle_center

def translation_vector(current_img, ref_img):
     ref_img = cv2.imread("goProHero10/src/Camera/images/linear/img_ref.png")
     
     _ , center_ref = find_circle(ref_img)

     if len(center_ref) == 0:
         raise ValueError("No circles found in the reference image")
     
     circle_pos_ref = center_ref[0]

     current_img = cv2.imread("goProHero10/src/Camera/images/linear/img_ref.png")
     _, center_curr = find_circle(current_img)

     if len(center_curr) == 0:
         raise ValueError("No circles found in the current image")
     
     circle_pos_curr = center_curr[0]

     t = np.array(circle_pos_curr) - np.array(circle_pos_ref)

     return t
    
     



ref = cv2.imread("goProHero10/src/Camera/images/linear/img_ref.png")
curr = cv2.imread("goProHero10/src/Camera/images/linear/img_ref.png")

t = translation_vector(curr, ref)

print("Translation vector: ", t)
