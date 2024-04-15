import cv2

# Read the image
image = cv2.imread("goProHero10/src/Camera/images/linear/img_0000.png")

# Convert the image to HSV color space
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for blue color in HSV
lower_blue = (90, 150, 150)  # Lower bound for blue
upper_blue = (120, 255, 255)  # Upper bound for blue

# Create a mask using the specified range
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
im = cv2.resize(result, (960, 540)) 
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
