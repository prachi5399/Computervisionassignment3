import cv2
import numpy as np
import matplotlib.pyplot as plt

pic_1 = cv2.imread('prachi1.png', 0) 
pic_2 = cv2.imread('prachi2.png', 0)

def ShowDisparity(bSize=5):
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize) 
    disparity = stereo.compute(pic_1, pic_2)
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min)) 
    return disparity

result = ShowDisparity(bSize=5) 
print(result) 
plt.imshow(result, 'gray') 
cv2.imwrite("output.png", result)
plt.axis('off')
plt.show()
