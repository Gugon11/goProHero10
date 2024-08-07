import numpy as np
import cv2
import glob
import pickle
import os

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

b = 7
a = 9
chessboardSize = (a, b)
frameSize = (640,480)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('/images/*.png')
images = glob.glob(os.path.join(os.getcwd(), 'goProHero10/src/Camera/img_cali', '*.png'))
images = images[:20]


savePath = os.path.join(os.getcwd(), "goProHero10/src/Camera/cali_chess_linear")
if (os.path.isdir(savePath) is not True):
    try:
        os.mkdir(savePath)
    except:
        pass
#end-if-else

newImagePrefix = "img_cal_"
newImageExtension = ".png"
maxLeadingZeros = 4


imgCounter = 0
for image in images:
    print(f"Reading {image} ...")
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        '''cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)'''
        
        #---------------
        filename = newImagePrefix + (maxLeadingZeros - len(str(imgCounter)))*"0"+str(imgCounter) + newImageExtension
        filePath = os.path.join(savePath, filename)
        
        cv2.imwrite(filePath, img)
        print("image saved!")
        imgCounter += 1
    else:
        print("Not found ...")

cv2.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

def save2file(filename, variable):
    if (".dat" not in filename): filename += ".dat"
    savePath = os.path.join(os.getcwd(), "goProHero10/src/Camera/camera_parameterslinear")
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


with open("cameraMatrix.dat", "+wb") as fid: np.save(fid, cameraMatrix)


print(f"ret: {ret}")
print(f"cameraMatrix: {cameraMatrix} \n")
print(f"distCoeffs: {distCoeffs}")

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

input(">>>")

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, distCoeffs), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(distCoeffs, open( "distCoeffs.pkl", "wb" ))


############## UNDISTORTION #####################################################

img = cv2.imread('goProHero10/src/Camera/img_cali/img_0000.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult1.png', dst)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
