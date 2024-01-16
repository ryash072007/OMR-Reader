import cv2
import numpy as np


def sharpen_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply unsharp masking to enhance edges
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Apply adaptive thresholding to remove background noise
    _, thresholded = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresholded

def get_image_part(contour_part, image):
    x, y, w, h = cv2.boundingRect(np.array(contour_part))
    return image[y : y + h, x : x + w]

def points(gray):
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

    #    Draw circles that are detected.
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
        ordered_circles.sort(
            key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0])
        )

        # Reverse the list because the points are sorted in counterclockwise order
        ordered_circles = ordered_circles[::-1]

        # Draw the circumference of the circle.
        cv2.circle(gray, (a, b), r, (100, 100, 0), 6)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(gray, (a, b), 1, (100, 0, 100), 4)
        cv2.imwrite("split/gray.png", gray)
        # cv2.imshow("Detected Circle", gray)
        # cv2.waitKey(0)

        # Define the four corners of the region of interest (ROI)
        roi_corners = np.array([(x, y) for (x, y) in ordered_circles], dtype=np.float32)

        # height, width = img.shape[0], img.shape[1]
        height, width = 3300, 2475

        # Define the destination points for the perspective transform
        dst_corners = np.array(
            [(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32
        )

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(roi_corners, dst_corners)

        # Apply the perspective transform to the image
        warped_img = cv2.warpPerspective(gray, M, (width, height))

        cv2.imwrite("split/warped_img.jpg", warped_img)


def get_points(sharpened_image):
    _, thresholded = cv2.threshold(sharpened_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_parts = list()
    largest_area = 0
    # Iterate over all contours
    for i, contour in enumerate(contours):
        # Get the image for the contour
        x, y, w, h = cv2.boundingRect(np.array(contour))
        if w * h > 8000:
            image_parts.append((x, y, w, h))
            if w * h > largest_area:
                largest_area = w * h


    circles_points = []

    for x, y, w, h in image_parts:
        image_part = sharpened_image[y : y + h, x : x + w]
        image_area = w * h
        circle_area = 3.14 * (min(w, h) / 2) ** 2
        filled_pixels = np.sum(image_part == 0)
        _range = range(int(0.95 * circle_area), int(1.05 * circle_area))
        # print(circle_area, filled_pixels)
        if filled_pixels in _range:
            circles_points.append((x, y, w, h))
    
    return circles_points


image = cv2.imread("images/test (3).jpg")
sharpened_image = sharpen_image(image)
circles_points = get_points(sharpened_image)
print(circles_points)

# cv2.waitKey(0)
cv2.destroyAllWindows()
