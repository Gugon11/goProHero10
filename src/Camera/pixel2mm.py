import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img):
    # Resize image to fit screen for better visualization
    im_resized = cv2.resize(img, (960, 540))
    cv2.imshow(title, im_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_img(image, x, y, h, w):
    cropped_img = image[y:y+h, x:x+w]
    return cropped_img

# Read image
img = cv2.imread("goProHero10/src/Camera/images/linear/img_0006.png")
img_crop = crop_img(img, 600, 80, 1000, 1200)

# Convert image to grayscale
img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
show_image("Grayscale Image", img_gray)

# Convert images to binary for paper detection
_, img_bin = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
show_image("Binary Image", img_bin)

# Opening binary image to remove noise around the paper
close_kernel = np.ones((9, 9), np.uint8)
img_dilated = cv2.dilate(img_bin, close_kernel, iterations=1)
img_closed = cv2.erode(img_dilated, close_kernel, iterations=1)
show_image("Opened Image", img_closed)

# Find contours
image_contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and aspect ratio
paper_contours = []
for contour in image_contours:
    area = cv2.contourArea(contour)
    if area < 1000:  # Ignore very small contours
        continue

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    # Check if the contour has an aspect ratio similar to A4 paper (210:297 ~ 0.707)
    if 0.6 < aspect_ratio < 0.85:
        paper_contours.append(contour)

# Draw all contours for debugging purposes
img_contours = cv2.drawContours(img_crop.copy(), image_contours, -1, (0, 255, 0), 2)
show_image("All Contours", img_contours)

# If no paper-like contours are found, raise an error
if not paper_contours:
    raise ValueError("No contours found with the expected area and aspect ratio")

# Find the largest contour among the filtered contours
largest_contour = max(paper_contours, key=cv2.contourArea)

# Draw the largest paper-like contour
img_largest_contour = cv2.drawContours(img_crop.copy(), [largest_contour], -1, (0, 0, 255), 2)
show_image("Largest Paper-Like Contour", img_largest_contour)

# Approximate the contour with a 4-sided polygon
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) != 4:
    raise ValueError("Could not find a quadrilateral. Found vertices: {}".format(len(approx)))
else:
    # Define the coordinates of the paper's vertices
    top_left = approx[0][0]
    top_right = approx[1][0]
    bottom_right = approx[2][0]
    bottom_left = approx[3][0]
    paper_coords = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Draw the approximated contour
    img_approx = cv2.drawContours(img_crop.copy(), [approx], -1, (255, 0, 0), 2)
    for point in paper_coords:
        cv2.drawMarker(img_approx, tuple(point.astype(int)), (0, 0, 255), markerType=cv2.MARKER_CROSS, 
                       markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    show_image("Approximated Contour", img_approx)

    # Define the coordinates of the rectangle's vertices
    dst_height = max(np.linalg.norm(top_left - bottom_left), np.linalg.norm(bottom_right - top_right))
    dst_width = dst_height * 297 / 210

    rectangle_coords = np.float32([
        top_left,
        top_left + [dst_width, 0],
        top_left + [dst_width, dst_height],
        top_left + [0, dst_height]
    ])

    # Calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(paper_coords, rectangle_coords)

    print("Transformation Matrix:")
    print(transform_matrix)

    pixel_to_mm = dst_height / 210.0
    print("Pixel to millimeter ratio =", pixel_to_mm, "pixels/mm")
