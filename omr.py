import cv2
import numpy as np

image = cv2.imread("images/omr-turned.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 205, 255, cv2.THRESH_BINARY)

circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=300, maxRadius=400)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)

while True:
    cv2.imshow("Image", image)
    if cv2.waitKey():
        break
