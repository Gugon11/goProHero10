import cv2
import pandas as pd

image=cv2.imread("goProHero10/src/Camera/images/img_0032.png")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Output", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
