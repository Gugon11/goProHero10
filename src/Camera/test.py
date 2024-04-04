import cv2
import pandas as pd

content = pd.read_csv('dist_coeffs.txt', delim_whitespace=True, header=None)
data = content.to_numpy()
print(data)
