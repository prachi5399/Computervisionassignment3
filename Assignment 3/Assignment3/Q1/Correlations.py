import cv2
import matplotlib.pyplot as plt

images = cv2.imread("capture_isp_2.png")

point_image = images[714:2228,341:1420]

test_images = [
    "capture_isp_1.png",
    "capture_isp_5.png",
    "capture_isp_3.png",
    "capture_isp_2.png",
    "capture_isp_10.png",
    "capture_isp_9.png",
    "capture_isp_6.png",
    "capture_isp_8.png",
    "capture_isp_4.png",
    "capture_isp_7.png"
]

correlations = []

for img in test_images:
    testImg = cv2.imread(img)
    croppedTestImg = testImg[714:2228,341:1420]
    plt.imshow(croppedTestImg)
    plt.show()
    X = croppedTestImg - point_image
    ssd = sum(X[:]**2)
    correlations.append(ssd)

print(correlations)


cv2.imshow("image of interest",images)
plt.imshow(point_image)
plt.show()