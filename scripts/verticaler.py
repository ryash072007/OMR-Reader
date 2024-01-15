import numpy as np
import cv2

omr_image = cv2.imread("images/omr_sheet -filled and circles.png")
circle_image = cv2.imread("images/circle_pointer.jpg", 0)

gray_omr_image = cv2.cvtColor(omr_image, cv2.COLOR_BGR2GRAY)
blurred_omr_image = cv2.GaussianBlur(gray_omr_image, (5, 5), 0)


matched_result = cv2.matchTemplate(
    blurred_omr_image, circle_image, cv2.TM_CCOEFF
)
locations = np.where(matched_result >= 0.9)

for pt in zip(*locations[::-1]):  # Swap columns and rows
    if pt[0] + circle_image.shape[1] > 1000 and pt[1] + circle_image.shape[0] > 1000:
        cv2.rectangle(
            omr_image,
            pt,
            (pt[0] + circle_image.shape[1], pt[1] + circle_image.shape[0]),
            (0, 0, 255),
            2,
        )


cv2.imwrite('split/matched_result.png', omr_image)
cv2.waitKey()
cv2.destroyAllWindows()
