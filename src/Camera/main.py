import os
import sys
import logging
import threading
import traceback
import pandas as pd
import numpy as np
import time

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
desiredFovName = "linear"

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
#Path to save the output images (when the user presses key 'p')
savePath = os.path.join(os.getcwd(), "goProHero10/src/Camera/img_cali")
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

#Reference image for translation vector calculation
img_ref = cv2.imread("goProHero10/src/Camera/images/linear/img_checkpoints.png")

#Ratio of pixel to millimeter obtained in pixel2mm.py
px2mm = 0.44812778376576956

# Initializing previous positions and times
previous_positions = {'blue': None, 'yellow': None, 'pink': None}
previous_times = {'blue': None, 'yellow': None, 'pink': None}

_, circle = goProCamera.find_circle(img_ref)
checkpoint_organized = np.array([circle[6],
                             circle[4],
                             circle[5],
                             circle[2],
                             circle[0],
                             circle[3],
                             circle[8],
                             circle[7]])

t_new = np.array([0, 0])


if (goProCamera.cap.isOpened() is False):
    logging.error("Unable to open the camera ...")
    raise Exception("Unable to open the camera ...")
else:
    logging.info("Camera was opened successfully.")
    
    while(goProCamera.cap.isOpened()):
        #Update the image on the frame:
        goProCamera.display()

        # Ensure the resolution and FOV are set to 1080p and linear
        if desiredResolutionName == "1080p" and desiredFovName == "linear":
            res, center = goProCamera.find_circle(goProCamera.frame)
        
            # Check if a circle was detected
            if len(center) == 0:
                center = np.array([0, 0]) #if no circle was detected, put center as [0, 0]
            
            t = center - circle[1]
            
            if not np.array_equal(t, -np.array([1636, 986])):  # If a valid center was detected
                t_new = t
            
            #Position of the origin
            origin_coords = circle[1] + t_new

            #Array of all the checkpoints with the translation vector
            checkPoints = checkpoint_organized + t_new
            
            centroids, res = goProCamera.detect_cars(res)

            # Get current time
            current_time = time.time()

            for color, centroid in zip(['blue', 'yellow', 'pink'], centroids):
                if centroid is not None and not np.array_equal(centroid, [0, 0]):
                    # Calculate velocity
                    if previous_positions[color] is not None and previous_times[color] is not None:
                        dt = current_time - previous_times[color]
                        velocity = goProCamera.calc_velocity(previous_positions[color], centroid, dt, px2mm)
                        print(f"{color.capitalize()} car velocity (cm/s): {velocity}")

                        # Calculate orientation
                        orientation = goProCamera.calc_orientation(previous_positions[color], centroid)
                        print(f"{color.capitalize()} car orientation (degrees): {orientation}")

                    # Update previous position and time
                    previous_positions[color] = centroid
                    previous_times[color] = current_time

                    #Convert centroid to real-world coordinates
                    centroid_origin = ((centroid - origin_coords)/px2mm)/10 #convert to cm
                    print(f"{color.capitalize()} car position (cm): {centroid_origin}")

            '''#Difference between car coordinates and origin coordinates
            centroids_origin_b = centroids[0] - origin_coords
            centroids_origin_y = centroids[1] - origin_coords
            centroids_origin_p = centroids[2] - origin_coords
            
            #Coordinates in cm
            origin_cm = (origin_coords/px2mm)/10 #centimeter
            centroids_cm_b = (centroids_origin_b/px2mm)/10 #centimeter
            centroids_cm_y = (centroids_origin_y/px2mm)/10
            centroids_cm_p = (centroids_origin_p/px2mm)/10



            print("Origin coords: ", origin_cm)
            print("Blue car: ", centroids_cm_b)
            print("Yellow car: ", centroids_cm_y)
            print("Pink car: ", centroids_cm_p)'''

        cv2.imshow(goProCamera.windowName, res)


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