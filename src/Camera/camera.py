import os
import numpy as np
import cv2
import pandas as pd

from utils.config import readConfig

import time
from utils.timeFunctions import countdown, currentTime
from utils.getCameras import get_available_cameras

#pip install pygrabber==0.1
from pygrabber.dshow_graph import FilterGraph
from UDPsender import UDPSender

class camera:
    def __init__(self, cameraConfig: str="camera", cameraName: str="GoPro Webcam", windowName: str="Frame") -> None:
        #------------------------------------------------------
        #Read the Configs
        self.config = readConfig(moduleName = cameraConfig)
        
        self.enableDirectShow = self.config["enableDirectShow"]
        self.showFPS          = self.config["showFPS"]
        self.imageSubfolder   = self.config["imageSubfolder"]
        self.frameWidth       = self.config["frameWidth"]
        self.frameHeight      = self.config["frameHeight"]
        #------------------------------------------------------
        #Create the Images Subfolder:
        self.savePath = os.path.join(os.getcwd(), "images")
        if (os.path.isdir(self.savePath) is not True):
            try:
                os.mkdir(self.savePath)
            except:
                pass
        #end-if-else
        #------------------------------------------------------
        #Get the Camera ID
        self.cameraID = self.getCameraID(goProName = cameraName)
        if self.cameraID == -1: raise Exception("'GoPro Webcam' was not found in the available cameras list.")
        
        #------------------------------------------------------
        #Start the Camera with/without Direct Shows
        #self.cap = cv2.VideoCapture(self.cameraID, cv2.CAP_DSHOW)
        if (self.enableDirectShow):
            self.cap = cv2.VideoCapture(self.cameraID, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.cameraID)

        #end-if-else
        #Add some delay to give time for the camera setup
        countdown(delay = 3, message="Opening Camera")
        #------------------------------------------------------
        if (self.showFPS):
            self.prev_frame_time = 0
            self.new_frame_time = 0
            self.fps = 0
        #end-if-else
        #------------------------------------------------------
        self.windowName = windowName
        
        self.frame = None
        self.positions_aruco = None
    #end-def


#-----------------------DETECTION WITH ARUCO MARKERS-------------------------------------------   
    def marker_position(self, tvec, rvec):
        """
        Calculate the position of the marker in the camera coordinate system.

        Parameters:
        tvec (numpy array): Translation vector of the marker.
        rvec (numpy array): Rotation vector of the marker.

        Returns:
        numpy array: Position of the marker in the camera coordinate system.
        """
        # Convert rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Invert rotation matrix to obtain the transformation from marker to camera
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    
        # Convert translation vector to a column vector
        tvec = tvec.reshape((3, 1))

        # Calculate marker position in camera coordinates using inverse transformation
        marker_pos = -np.dot(rotation_matrix_inv, tvec)

        return marker_pos
#-------------------------------------------------------------------------------------------------
    def pose_estimation(self, frame, aruco_dict, matrix_coefficients, distortion_coefficients):
        ret, frame = self.cap.read()
        matrix_reader = pd.read_csv('camera_matrixlinear.txt', delim_whitespace=True, header=None)
        matrix_coefficients = matrix_reader.to_numpy()
        dist_reader = pd.read_csv('dist_coeffslinear.txt', delim_whitespace=True, header=None)
        distortion_coefficients= dist_reader.to_numpy()
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        positions_aruco = []  # List to store positions of all detected markers
        
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
                
                # Calculate marker position
                position_aruco = self.marker_position(tvec, rvec)
                
                # Append position to the list
                positions_aruco.append(position_aruco)
        
        self.positions_aruco = positions_aruco
                

        return frame, positions_aruco


#-------------------------------DETECTION WITH CAR's COLOR-----------------------------------------------
    def crop_img(self, image, x, y, h, w):
        cropped_img = image[y:y+h, x:x+w]
        return cropped_img
    
    
    def detect_cars(self, image):
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

            # Extract the centroid of the largest contour
            centroid = None
            max_contour_area = 0
            for cnt in filtered_contours:
                area = cv2.contourArea(cnt)
                if area > max_contour_area:
                    moments = cv2.moments(cnt)
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        centroid = (cX, cY)
                        max_contour_area = area
            
            # Draw the centroid on the result image
            if centroid is not None:
                cv2.circle(result, centroid, 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return centroid, result

        def detect_yellow_cars(image):
            # Convert the image to HSV color space
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for yellow color in HSV
            
            lower_yellow = np.array([20, 100, 100])   # Lower bound for yellow
            upper_yellow = np.array([30, 255, 255])   # Upper bound for yellow


            # Create masks using the specified ranges
            mask1 = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
            #mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)


            # Perform morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            min_area = 500
            max_area = 1500
            filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

            # Draw the filtered contours on the original image
            result = image.copy()
            cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

            # Extract the centroid of the largest contour
            centroid = None
            max_contour_area = 0
            for cnt in filtered_contours:
                area = cv2.contourArea(cnt)
                if area > max_contour_area:
                    moments = cv2.moments(cnt)
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        centroid = (cX, cY)
                        max_contour_area = area

            # Draw the centroid on the result image
            if centroid is not None:
                cv2.circle(result, centroid, 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return centroid, result

        def detect_pink_cars(image):
            #Convert the image to HSV color space
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define the lower and upper bounds for pink color in HSV
            lower_pink1 = np.array([150, 50, 50])
            upper_pink1 = np.array([170, 255, 255])

            lower_pink2 = np.array([0, 50, 50])
            upper_pink2 = np.array([10, 255, 255])

            # Create masks using the specified ranges
            mask1 = cv2.inRange(image_hsv, lower_pink1, upper_pink1)
            mask2 = cv2.inRange(image_hsv, lower_pink2, upper_pink2)

            # Combine masks
            mask = cv2.bitwise_or(mask1, mask2)

            # Perform morphological operations to clean up the mask
            kernel = np.ones((7, 7), np.uint8)  # Increase kernel size for better noise reduction
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Additional opening to remove smaller noise

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            min_area = 500  # Adjust minimum area threshold
            max_area = 2500  # Adjust maximum area threshold
            filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

            # Draw the filtered contours on the original image
            result = image.copy()
            cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

            # Extract the centroid of the largest contour
            centroid = None
            max_contour_area = 0
            for cnt in filtered_contours:
                area = cv2.contourArea(cnt)
                if area > max_contour_area:
                    moments = cv2.moments(cnt)
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        centroid = (cX, cY)
                        max_contour_area = area
            
            # Draw the centroid on the result image
            if centroid is not None:
                cv2.circle(result, centroid, 10, (255, 255, 255), -1)
                cv2.putText(result, "centroid", (centroid[0] - 25, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return centroid, result


        def adjust_intensity(image, alpha=0.35, beta=0):
            # Perform intensity adjustment using alpha blending
            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted_image

        # Detect blue cars
        blue_centroids, blue_result = detect_blue_cars(image)

        # Adjust intensity of blue result
        blue_result_adjusted = adjust_intensity(blue_result)

        # Detect red cars
        yellow_centroids, yellow_result = detect_yellow_cars(image)

        # Adjust intensity of red result
        yellow_result_adjusted = adjust_intensity(yellow_result)

        # Detect pink cars
        pink_centroids, pink_result = detect_pink_cars(image)

        # Adjust intensity of pink result
        pink_result_adjusted = adjust_intensity(pink_result)

        # Combine adjusted results
        combined_result = cv2.add(cv2.add(blue_result_adjusted, yellow_result_adjusted), pink_result_adjusted)

        # Combine centroids
        all_centroids = [blue_centroids, yellow_centroids, pink_centroids]

        return all_centroids, combined_result


    def find_circle(self, img):
        # Convert image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #show_image("Gray image", gray)

        # Adaptive thresholding for better binary conversion
        img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #show_image("Binary Image", img_bin)

        # Opening binary image to remove noise around the circles
        close_kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img_bin, close_kernel, iterations=1)
        img_closed = cv2.erode(img_dilated, close_kernel, iterations=1)
        #show_image("Opened Image", img_closed)

        # Find circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            img_closed, 
            cv2.HOUGH_GRADIENT, 
            dp=1.0, 
            minDist=400, # Adjusted minimum distance between circles
            param1=50, 
            param2=30, 
            minRadius=10,
            maxRadius=30
        )

        # List to store the centers of the circles
        centers = []

        # If circles are detected, draw them on the original image and store the centers
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                centers.append(center)
                # Draw the outer circle
                cv2.circle(img, center, i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(img, center, 2, (0, 0, 255), 3)
        
        return img, centers
#---------------------------------------------Display Live Frame-----------------------------------------
    
    def display(self):
        ret, self.frame = self.cap.read()
        self.changeFrameSize()
        if self.showFPS:
            self.new_frame_time = currentTime()
            # fps will be number of frame processed in given time frame 
            # since their will be most of time error of 0.001 second 
            # we will be subtracting it to get more accurate result 
            
            fps = 1/(self.new_frame_time - self.prev_frame_time) 
            self.prev_frame_time = self.new_frame_time

            self.fps = int(fps)
            # converting the fps to string so that we can display it on frame by using putText function 
            fps = str(self.fps)
            
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(self.frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        #end-if-else

        '''matrix_reader = pd.read_csv('camera_matrixlinear.txt', delim_whitespace=True, header=None)
        k = matrix_reader.to_numpy()
        dist_reader = pd.read_csv('dist_coeffslinear.txt', delim_whitespace=True, header=None)
        d = dist_reader.to_numpy()
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        output, _ = self.pose_estimation(self.frame, aruco_dict, k, d)'''

        udpsender = UDPSender()

        initial_center = None
        racetrack = self.crop_img(self.frame, 600, 80, 1000, 1200)

        #Check if center was detected. If it was, it uses that value for the rest of the live video
        if initial_center is None:
            racetrack, initial_center =self.find_circle(racetrack)
        else:
            center = initial_center
        
        print("Circle Center", center)
        
        centroids, res = self.detect_cars(racetrack)
        print("Cars position", centroids)

        #Distance of each car to the origin in pixels
        blue_px = centroids[0]-center
        yellow_px = centroids[1]-center
        pink_px = centroids[2]-center

        #Ratio of pixel to millimeter obtained in pixel2mm.py
        px2mm = 0.6337807227544455

        #Distance of each car to the origin in mm
        blue_mm = blue_px/px2mm
        yellow_mm = yellow_px/px2mm
        pink_mm = pink_px/px2mm

        if res is None or res.size == 0:
            print("Detection result is empty")
            return

        cv2.imshow(self.windowName, res)

        while(camera.display()):
            udpsender.send_data("blue", blue_mm, 45.0)
            udpsender.send_data("yellow", yellow_mm, 90.0)
            udpsender.send_data("pink", pink_mm, 135.0)
            time.sleep(0.01667)  # Send data at 60 Hz


        

        key = cv2.waitKey(1)
        if key == 27: #ESC Key to exit
            pass
        
    #end-def
    
    
    def changeFrameSize(self) -> None:
        cv2.namedWindow(self.windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 
                         self.frameWidth,
                         self.frameHeight)
        return
    #end-def
    
    def getCameraID(self, goProName: str="") -> int:
        cameras = self.get_available_cameras()
        for cameraID, cameraName in cameras.items():
            if (cameraName.lower() == goProName.lower()):
                return cameraID
            else:
                pass
            #end-if-else
        #end-for
        return -1
    #end-def
    
    def get_available_cameras(self) -> dict:
        #https://stackoverflow.com/questions/70886225/get-camera-device-name-and-port-for-opencv-videostream-python
        devices = FilterGraph().get_input_devices()

        available_cameras = {}

        for device_index, device_name in enumerate(devices):
            available_cameras[device_index] = device_name
        #end-for
        
        print(available_cameras)
        return available_cameras
    #end-def
    
#end-class

if __name__ == "__main__":
    pass
#end-if-else