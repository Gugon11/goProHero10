import os
import sys
import logging
import threading
import traceback
import pandas as pd
import numpy as np

import cv2
from cv2 import aruco
from camera import camera
from goProHero10 import goProHero10

from utils import logger
from utils.config import readConfig
from utils.timeFunctions import countdown


#----------------------------------------------------------------
config = readConfig(moduleName=__name__)

#----------------------------------------------------------------
#Initialize the Logger
logger = logger.setupLogger()

#----------------------------------------------------------------
#Initialize the API Handler
logging.info("Initializing the goProHero10 [API] ...")
gopro = goProHero10()
logging.info(gopro.showConnectionDetails())

#----------------------------------------------------------------
#Adjust Resolution + Fov
desiredResolutionName = "1080p"
desiredFovName = "wide"

desiredResolution = gopro.resolutions[desiredResolutionName] #1080p,720p,420p
desiredFov = gopro.fovs[desiredFovName] #Wide, Narrow, Superview, Linear

gopro.startWebcam["params"]["res"] = desiredResolution
gopro.startWebcam["params"]["fov"] = desiredFov


# frameSizes = {'480p': ( 640,  480),
#               '720p': (1280,  720),
#               '1080p': (1920, 1080)}
frameSizes = {'480p': ( 640,  480),
              '720p': (1280,  720),
              '1080p': (1280, 720)} #reduce 1080p window size

#----------------------------------------------------------------
#Check if camera is in "Idle" Mode
#When the camera is in this mode, start the webcam with the
#desired RESOLUTION and FOV
currentStatus = ""
while (currentStatus != "idle"):
    currentStatus = gopro.getCameraState()
    logging.info(f"Current Camera Status: {currentStatus}.")
    if (currentStatus == "idle"):
        logging.info(f'Starting webcam with:\nurl: {gopro.startWebcam["url"]}\nparams: {gopro.startWebcam["params"]}')
        logging.info(f'Resolution: {desiredResolution}\nFOV:{desiredFov}\n')
        gopro.getHTTP(paramURL  = gopro.startWebcam["url"],
                      parameters= gopro.startWebcam["params"])
    else:
        logging.info("Camera is not available for webcam mode ...")
        
        logging.info('Forcing Webcam exit ({gopro.exitWebcamPreview["url"]}) ...')
        gopro.getHTTP(paramURL = gopro.exitWebcamPreview["url"])
        countdown(delay=3, message="Closing Webcam preview ...")
        
        logging.info("Sending Keep Alive signal ...")
        gopro.keepAliveSignal()
    #end-if-else
#end-while

newCurrentStatus = gopro.getCameraState()
logging.info(f"Current Camera Status: {currentStatus}.")
print(f"Current Camera Status: {currentStatus}.")

#----------------------------------------------------------------

'''
def keyPressHandler():
    not used
    while True:
        try:
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                del goProCamera
            elif (cv2.waitKey(1) & 0xFF == ord('1')):
                gopro.startWebcam["params"]["res"] = gopro.resolutions["1080p"]
            elif (cv2.waitKey(1) & 0xFF == ord('2')):
                gopro.startWebcam["params"]["res"] = gopro.resolutions["720p"]
            elif (cv2.waitKey(1) & 0xFF == ord('3')):
                gopro.startWebcam["params"]["res"] = gopro.resolutions["480p"]
            else:
                pass
            #end-if-else
        except:
            pass
        #end-try-except
        '''
#end-def

#threadKeyPress = threading.Thread(target = keyPressHandler, args=[])
#threadKeyPress.start()
#----------------------------------------------------------------
'''def pose_estimation(frame, aruco_dict, matrix_coefficients, distortion_coefficients):
    matrix_reader = pd.read_csv('camera_matrix.txt', delim_whitespace=True, header=None)
    matrix_coefficients = matrix_reader.to_numpy()
    dist_reader = pd.read_csv('dist_coeffs.txt', delim_whitespace=True, header=None)
    distortion_coefficients= dist_reader.to_numpy()
    aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)
    
    rvecs = []
    tvecs = []
    
    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            
            rvecs.append(rvec)
            tvecs.append(tvec)

    return frame, rvecs, tvecs'''

def marker_position(tvec, rotation_matrix):
    """
    Calculate the position of the marker in the camera coordinate system.

    Parameters:
        tvec (numpy array): Translation vector of the marker.
        rotation_matrix (numpy array): Rotation matrix of the marker.

    Returns:
        numpy array: Position of the marker in the camera coordinate system.
    """
    # Invert rotation matrix to obtain the transformation from marker to camera
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    
    # Convert translation vector to a column vector
    tvec = tvec.reshape((3, 1))

    # Calculate marker position in camera coordinates using inverse transformation
    marker_pos = -np.dot(rotation_matrix_inv, tvec)

    return marker_pos
#----------------------------------------------------------------
#Path to save the output images (when the user presses key 'p')
savePath = os.path.join(os.getcwd(), "images")
if (os.path.isdir(savePath) is not True):
    try:
        os.mkdir(savePath)
    except:
        pass
#end-if-else

newImagePrefix = "img_"
newImageExtension = ".png"
maxLeadingZeros = 4 #example: img_0000.png, img_0001.png, ...
imgCounter = 0

#----------------------------------------------------------------


#Initialize the Camera/Preview/Player Handler
logging.info("Initializing the Camera [openCV] ...")
goProCamera = camera()

#Set the Frame (width, height) according to the resolution
goProCamera.frameWidth, goProCamera.frameHeight = frameSizes[desiredResolutionName]
#Set the Window Name:
goProCamera.windowName = desiredResolutionName + " | " + desiredFovName 




changeParamsFlag = False


if (goProCamera.cap.isOpened() is False):
    logging.error("Unable to open the camera ...")
    raise Exception("Unable to open the camera ...")
else:
    logging.info("Camera was opened successfully.")
    
    while(goProCamera.cap.isOpened()):
        #Update the image on the frame:
        goProCamera.display()
        
        position_ArUco = marker_position(goProCamera.tvecs, goProCamera.rvecs)
        print(position_ArUco)
        #Check the Key presses:
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        elif (cv2.waitKey(1) & 0xFF == ord('a')):
            desiredResolutionName = "1080p"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('s')):
            desiredResolutionName = "720p"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('d')):
            desiredResolutionName = "480p"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('z')):
            desiredFovName = "wide"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('x')):
            desiredFovName = "narrow"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('c')):
            desiredFovName = "superview"
            changeParamsFlag = True
        elif (cv2.waitKey(1) & 0xFF == ord('v')):
            desiredFovName = "linear"
            changeParamsFlag = True
            
        elif cv2.waitKey(5) == ord('p'):
            filename = newImagePrefix + (maxLeadingZeros - len(str(imgCounter)))*"0"+str(imgCounter) + newImageExtension
            filePath = os.path.join(savePath, filename)
            cv2.imwrite(filePath, goProCamera.frame)
            logging.info(f"Image {filename} saved!")
            imgCounter += 1
        else:
            pass
        #end-if-else
        
        #Update/Change the Resolution/FOV upon user's request
        if (changeParamsFlag):
            msg = f"Received user request to change the resolution to {desiredResolutionName}."
            logging.info(msg)
            print(msg)
            
            desiredResolution = gopro.resolutions[desiredResolutionName]
            desiredFov = gopro.fovs[desiredFovName]
            
            gopro.startWebcam["params"]["res"] = desiredResolution
            gopro.startWebcam["params"]["fov"] = desiredFov
            
            logging.info(f'Starting webcam with:\nurl: {gopro.startWebcam["url"]}\nparams: {gopro.startWebcam["params"]}')
            logging.info(f'Resolution: {desiredResolution}\nFOV:{desiredFov}\n')
            #gopro.getHTTP(paramURL = gopro.exitWebcamPreview["url"])
            gopro.getHTTP(paramURL = gopro.stopWebcam["url"])
            #countdown(delay=1, message="Closing Webcam preview and restarting...")
            gopro.getHTTP(paramURL  = gopro.startWebcam["url"],
                          parameters= gopro.startWebcam["params"])
            
            #Adjust the frame size according to the resolution
            goProCamera.frameWidth, goProCamera.frameHeight = frameSizes[desiredResolutionName]
            
            
            #Adjust the Window Name:
            previousWindowName = goProCamera.windowName
            goProCamera.windowName = desiredResolutionName + " | " + desiredFovName 
            
            cv2.destroyWindow(previousWindowName)
            
            changeParamsFlag = False
        #end-if-else
        
    #end-while
#end-if-else


goProCamera.cap.release()

#----------------------------------------------------------------


#threadKeyPress.stop()

logging.info("End of program.")