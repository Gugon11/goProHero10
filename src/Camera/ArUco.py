import os
import cv2

newImagePrefix = "aruco_"
newImageExtension = ".png"
maxLeadingZeros = 4  # example: img_0000.png, img_0001.png, ...
imgCounter = 0
# Generate ArUco Marker
def generate_ArUco(marker_id, side_pixels):
    # Define parameters for the ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Define the dictionary for ArUco markers

    # Generate the ArUco marker image
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_pixels)
    
    # Save the generated ArUco marker image
    global imgCounter
    img_name = f"{newImagePrefix}{str(imgCounter).zfill(maxLeadingZeros)}{newImageExtension}"
    img_path = os.path.join(saveAruco, img_name)
    cv2.imwrite(img_path, marker_image)
    saved = cv2.imwrite(img_path, marker_image)
    if saved:
        print(f"Image saved successfully: {img_path}")
    else:
        print(f"Failed to save image: {img_path}")
    imgCounter += 1

# Save to path generated arucos
saveAruco = os.path.join(os.getcwd(), "goProHero10\src\Camera\ArUco_img")
if not os.path.isdir(saveAruco):
    try:
        os.mkdir(saveAruco)
    except Exception as e:
        print(f"Error creating directory: {e}")

generate_ArUco(36, 400)
generate_ArUco(50,400)
generate_ArUco(80, 300)
