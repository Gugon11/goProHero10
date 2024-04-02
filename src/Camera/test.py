import cv2


# Load the image
image = cv2.imread("goProHero10/src/Camera/images/img_0032.png")

# Check if the image was successfully loaded
if image is not None:
    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # Wait for any key to be pressed
    cv2.destroyAllWindows()  # Close all windows
else:
    print("Image not found or could not be loaded.")
