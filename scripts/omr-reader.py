import cv2
import numpy as np


def get_contour_image(contours, hierarchy, index, shape):
    # Create an empty image to draw the contours on
    contour_image = np.zeros(shape, dtype=np.uint8)

    # Draw the contour at the given index
    contour_image = cv2.drawContours(
        contour_image, contours, index, (100), thickness=cv2.FILLED
    )

    # Get the index of the first child of the contour
    child_index = hierarchy[0][index][2]

    # Draw all child contours
    while child_index != -1:
        contour_image = cv2.drawContours(
            contour_image, contours, child_index, (255), thickness=cv2.FILLED
        )
        # Get the next child contour
        child_index = hierarchy[0][child_index][0]

    return contour_image


def get_image_part(contour_part, image):
    x, y, w, h = cv2.boundingRect(np.array(contour_part))
    return image[y : y + h, x : x + w]


def get_omr_area(thresholded, contours):
    omr_area = None
    for i, contour in enumerate(contours):
        # Perform necessary operations on each contour (e.g., bounding rectangle, aspect ratio, etc.)
        x, y, w, h = cv2.boundingRect(contour)

        # Perform OMR logic based on the extracted information
        # cv2.imwrite(f"split/processed_image_{i}.jpg", image[y: y + h, x: x + w])

        if not omr_area:
            omr_area = (x, y, w, h)
        else:
            if w * h > omr_area[2] * omr_area[3]:
                omr_area = (x, y, w, h)

    x, y, w, h = omr_area
    omr_area = thresholded[y : y + h, x : x + w]
    return omr_area


def get_areas_in_omr(omr_area, contours):
    images = []
    for i, contour in enumerate(contours):
        # Perform necessary operations on each contour (e.g., bounding rectangle, aspect ratio, etc.)
        x, y, w, h = cv2.boundingRect(contour)
        images.append(omr_area[y : y + h, x : x + w])
    return images


def read_omr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image (e.g., convert to grayscale, apply thresholding, etc.)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Perform OMR processing (e.g., detect contours, analyze regions, etc.)
    image_contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Process each contour and extract relevant information
    omr_area = get_omr_area(thresholded, image_contours)
    # cv2.imwrite("split/omr_area.jpg", omr_area)

    #
    # Get contours and hierarchy
    omr_area_contours, _hierarchy = cv2.findContours(
        omr_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the contour image for the first contour and its children
    omr_area_contours_reversed = list(reversed(omr_area_contours))
    part_1 = get_image_part(omr_area_contours_reversed[2], omr_area)
    part_2 = get_image_part(omr_area_contours_reversed[4], omr_area)
    part_3 = get_image_part(omr_area_contours_reversed[6], omr_area)

    cv2.imwrite("split/omr_part_1.jpg", part_1)
    cv2.imwrite("split/omr_part_2.jpg", part_2)
    cv2.imwrite("split/omr_part_3.jpg", part_3)
    
    

    # Now contour_image contains the first contour and all its child contours
    # cv2.imwrite(f"split/omr_area_area.jpg", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage example
image_path = "images/omr_sheet - filled.jpg"
read_omr(image_path)
