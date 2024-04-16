import numpy as np
import time
import cv2


# read the image
image = cv2.imread("img1.jpg")
# convert from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


#For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
"""
Hue is responsible for modeling the color type, 
saturation we could understand as the strength with which the color is displayed and finally 
value is the brightness or intensity of the color
"""




#RED:




for h in range(0, 255):
    #h = 160
    s = 80
    v = 150
    lower_limit = np.array([h, s, v])     # Lower HSV values for green
    upper_limit = np.array([170, 255, 255])   # Upper HSV values for green
    # create a mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask= mask)
    


    #image = cv2.resize(image, (960, 540))
    #hsvImage = cv2.resize(hsv_image, (960, 540))
    mask = cv2.resize(mask, (960, 540))
    res = cv2.resize(res, (960, 540))

    #cv2.imshow('frame',image)
    #cv2.imshow('hsvImage',hsvImage)

    print("h="+str(h)+"|"+"s="+str(s)+"|"+"v="+str(v))
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    cv2.waitKey(50)
    
#end-for

