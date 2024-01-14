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


def get_index_with_highest_white(image_list):
    index = -1
    max_white_pixels = 0
    for i, img in enumerate(image_list):
        # Count the number of white pixels in the image
        white_pixels = np.sum(img == 255)
        # print(white_pixels)
        # If this image has more white pixels than the current maximum, update the index and max_white_pixels
        if white_pixels > 700 and white_pixels > max_white_pixels:
            index = i
            max_white_pixels = white_pixels

    match index:
        case 0:
            index = 'A'
        case 1:
            index = 'B'
        case 2:
            index = 'C'
        case 3:
            index = 'D'
        case _:
            index = 'None Selected'
    
    return index

def split_image_times(image, h_times, v_times):
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)

    # Split the image
    images = []
    for i in range(h_times):
        row = []
        for j in range(v_times):
            # Adjust the last slice to include any remaining pixels
            h_end = image.shape[0] if i == h_times - 1 else (i + 1) * h_size
            v_end = image.shape[1] if j == v_times - 1 else (j + 1) * v_size
            row.append(image[i * h_size : h_end, j * v_size : v_end])
        images.append(row)

    return images


def split_image_times_list(image, h_times, v_times):
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)

    # Split the image
    images = []
    for i in range(h_times):
        for j in range(v_times):
            # Adjust the last slice to include any remaining pixels
            h_end = image.shape[0] if i == h_times - 1 else (i + 1) * h_size
            v_end = image.shape[1] if j == v_times - 1 else (j + 1) * v_size
            images.append(image[i * h_size : h_end, j * v_size : v_end])

    return images


def remove_top_areas(image):
    for y in range(image.shape[1]):
        if np.any(image[y] == 255):
            break
    return image[y:]


def remove_bottom_areas(image):
    for y in range(-1, -image.shape[1] - 1, -1):
        if np.any(image[y] == 255):
            break
    return image[:y]


def remove_left_areas(image):
    # Iterate over each column from left to right
    for x in range(image.shape[0]):
        # If any pixel in this column is white, break the loop
        if np.any(image[:, x] == 255):
            break

    # Return the image from the first white pixel column to the right
    return image[:, x:]


def remove_right_areas(image):
    # Iterate over each column from left to right
    for x in range(-1, -image.shape[0] - 1, -1):
        # If any pixel in this column is white, break the loop
        if np.any(image[:, x] == 255):
            break

    # Return the image from the first white pixel column to the right
    return image[:, :x]


def remove_excess_areas(image):
    top_removed = remove_top_areas(image)
    bottom_removed = remove_bottom_areas(top_removed)
    left_removed = remove_left_areas(bottom_removed)
    right_removed = remove_right_areas(left_removed)
    return right_removed
    # cv2.imshow("", right_removed)
    # cv2.waitKey()


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

    # cv2.imwrite("split/omr_part_1.jpg", part_1)
    # cv2.imwrite("split/omr_part_2.jpg", part_2)
    # cv2.imwrite("split/omr_part_3.jpg", part_3)

    part_1_images = split_image_times(part_1, 4, 1)
    part_2_images = split_image_times(part_2, 4, 1)
    part_3_images = split_image_times(part_3, 4, 1)
    
    images = [part_1_images, part_2_images, part_3_images]
    index = 1
    answers = {}
    for a in range(3):
        for b in range(4):
            block = images[a][b][0]
            block = block[
                int(0.05 * block.shape[1]) : int(0.95 * block.shape[1]),
                int(0.01 * block.shape[0]) : -int(0.01 * block.shape[0]),
            ]
            # cv2.imshow("", block)
            # cv2.waitKey()
            columns = split_image_times(block, 1, 4)
            columns = columns[0]
            # print(len(columns))
            # for i in columns:
            #     cv2.imshow("", i)
            #     cv2.waitKey()
            # print(columns)

            proper_images = []
            for i, img in enumerate(columns):
                if img.size > 1500:
                    processed = remove_excess_areas(img)
                    proper_images.append(processed)
            
            image_list = [[],[],[],[],[]]
            
            for img in proper_images:
                split_images = split_image_times_list(img, 5, 1)
                for i, split_img in enumerate(split_images):
                    image_list[i].append(split_img)
                    # cv2.imwrite(f"split/split_{index}_{i}.jpg", split_img)
                index += 1

            for lst in image_list:
                highest_index = get_index_with_highest_white(lst)
                answers[len(answers) + 1] = highest_index
    cv2.destroyAllWindows()
    print(answers)

# Usage example
image_path = "images/omr_sheet - filled.jpg"
read_omr(image_path)
