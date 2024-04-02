import numpy as np
import cv2
import glob
import pickle
import os
from glob import glob

# Define the size of the chessboard pattern used for calibration
pattern_size = (7, 9)

# Termination criteria for corner sub pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create arrays to store object points (3D points) and image points (2D points)
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Define the real world coordinates of the chessboard pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Directory containing calibration images
calibration_dir = "goProHero10/src/Camera/images"
calibration_images = glob(calibration_dir + "/*.png")
print(len(calibration_images))

for img_path in calibration_images:
    calibration_image = cv2.imread(img_path)
    gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners and refine them
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        #Draw and display the corners
        cv2.drawChessboardCorners(calibration_image, pattern_size, corners2, ret)
        cv2.imshow('',calibration_image)
        cv2.waitKey(1000)
    
    else:
        print(f"Chessboard corners not found in {img_path}")
#print(len(objpoints))
#print(len(imgpoints))
# Calibrate the camera
if objpoints and imgpoints:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Print calibration matrices
    print("Intrinsic Matrix (K):")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs)
    # Save camera matrix and distortion coefficients to text files
    np.savetxt('camera_matrix.txt', camera_matrix)
    np.savetxt('dist_coeffs.txt', dist_coeffs)


else:
    print("Insufficient data for calibration.")

'''
def save2file(filename, variable):
    if (".dat" not in filename): filename += ".dat"
    savePath = os.path.join(os.getcwd(), "goProHero10\src\Camera\camera_parameters")
    filePath = os.path.join(savePath, filename)
    if (os.path.isdir(savePath) is not True):
        try:
            os.mkdir(savePath)
        except:
            pass
    #end-if-else
    with open(filePath, "w+") as fid: fid.write(str(variable))
    return
#end-def
'''

    

'''
cameraName = "C1_"

fx = cameraMatrix[0][0]
ox = cameraMatrix[0][2]
fy = cameraMatrix[1][1]
oy = cameraMatrix[1][2]

print("fx = ", fx)
print("fy = ", fy)
print("ox = ", ox)
print("oy = ", oy)

save2file(cameraName+"fx", fx)
save2file(cameraName+"fy", fy)
save2file(cameraName+"ox", ox)
save2file(cameraName+"oy", oy)


with open("goProHero10\src\Camera\cameraMatrix.dat", "+wb") as fid: np.save(fid, cameraMatrix)


print(f"ret: {ret}")
print(f"cameraMatrix: {cameraMatrix} \n")
print(f"distCoeffs: {ret}")

print("\nRVecs:")
for rvec, i in zip(rvecs, range(0,len(rvecs))):
    string = ""
    for z in rvec: string += str(z)
    print(string)
#print(f"rotation vectors: {rvecs}")

print("\nTVecs")
for tvec in tvecs:
    string = ""
    for z in tvec: string += str(z)
    print(string)
#print(f"translation vectors: {tvecs}")

input(">>>")'''

"""
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))


############## UNDISTORTION #####################################################

img = cv.imread('cali5.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
"""