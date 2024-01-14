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


def split_image_times_row(image, h_times, v_times):
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)

    # Split the image
    images = []
    for i in range(0, image.shape[0], h_size):
        row = []
        for j in range(0, image.shape[1], v_size):
            row.append(image[i : i + h_size, j : j + v_size])
        images.append(row)

    return images


def split_image_times_list(image, h_times, v_times):
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)
    # Split the image
    images = []
    for i in range(0, image.shape[0], h_size):
        for j in range(0, image.shape[1], v_size):
            images.append(image[i : i + h_size, j : j + v_size])

    return images


def split_image_percentage(image, proportion):
    # Calculate the split position
    split_position = int(image.shape[0] * proportion)

    # Split the image
    image1 = image[:split_position, :]
    image2 = image[split_position:, :]

    return image1, image2


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

    #
    # Get contours and hierarchy
    omr_area_contours, _hierarchy = cv2.findContours(
        omr_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    _, question_area = split_image_percentage(omr_area, 1 - 0.955)
    part_1, part_2, part_3 = split_image_times_list(question_area, 1, 3)

    part_1_parts = split_image_times_list(part_1, 4, 1)
    part_1_1, part_1_2, part_1_3, part_1_4 = (
        part_1_parts[0],
        part_1_parts[1],
        part_1_parts[2],
        part_1_parts[3],
    )

    part_1_1_parts = split_image_times_list(part_1_1, 1, 5)
    (
        part_1_1_parts_1,
        part_1_1_parts_2,
        part_1_1_parts_3,
        part_1_1_parts_4,
        part_1_1_parts_5,
    ) = (
        part_1_1_parts[0],
        part_1_1_parts[1],
        part_1_1_parts[2],
        part_1_1_parts[3],
        part_1_1_parts[4],
    )



    cv2.imwrite("split/part_1_1_parts_1.jpg", part_1_1_parts_1)
    cv2.imwrite("split/part_1_1_parts_2.jpg", part_1_1_parts_2)
    cv2.imwrite("split/part_1_1_parts_3.jpg", part_1_1_parts_3)
    cv2.imwrite("split/part_1_1_parts_4.jpg", part_1_1_parts_4)
    cv2.imwrite("split/part_1_1_parts_5.jpg", part_1_1_parts_5)
    # cv2.imwrite("split/part_1_2.jpg", part_1_2)
    # cv2.imwrite("split/part_1_3.jpg", part_1_3)
    # cv2.imwrite("split/part_1_4.jpg", part_1_4)

    # Get the contour image for the first contour and its children
    # omr_area_contours_reversed = list(reversed(omr_area_contours))
    # part_1 = get_image_part(omr_area_contours_reversed[2], omr_area)
    # part_2 = get_image_part(omr_area_contours_reversed[4], omr_area)
    # part_3 = get_image_part(omr_area_contours_reversed[6], omr_area)

    cv2.destroyAllWindows()


# Usage example
image_path = "images/omr_sheet - filled.jpg"
read_omr(image_path)
