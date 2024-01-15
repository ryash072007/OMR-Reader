import cv2
import numpy as np

# Read image.
img = cv2.imread("images/omr_sheet -filled and circles.png", cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    1,
    20,
    param1=50,
    param2=30,
    minRadius=1,
    maxRadius=40,
)

# Draw circles that are detected.
if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    circles = []
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        circles.append((r, (a, b)))

    circles.sort(reverse=True)
    circles = circles[:4]
    
    ordered_circles = [i[1] for i in circles]
    
    center = np.mean(ordered_circles, axis=0)

    # Sort the points by the angle they make with the center
    ordered_circles.sort(key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

    # Reverse the list because the points are sorted in counterclockwise order
    # ordered_circles = ordered_circles[::-1]

    # # Draw the circumference of the circle.
    # cv2.circle(img, (a, b), r, (0, 100, 0), 2)

    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(img, (a, b), 1, (0, 0, 100), 3)
    # # cv2.imshow("Detected Circle", img)
    # # cv2.waitKey(0)


    # Define the four corners of the region of interest (ROI)
    roi_corners = np.array([(x, y) for (x, y) in ordered_circles], dtype=np.float32)

    height, width = img.shape[0], img.shape[1]
    # height, width = img.shape[0], img.shape[1]

    # Define the destination points for the perspective transform
    dst_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(roi_corners, dst_corners)

    # Apply the perspective transform to the image
    warped_img = cv2.warpPerspective(img, M, (width, height))

    cv2.imwrite("split/warped_img.jpg", warped_img)
