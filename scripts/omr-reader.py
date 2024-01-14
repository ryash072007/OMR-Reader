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
    index = 1
    for a in range(2):
        for b in range(1):
            block = part_1_images[a][b]
            block = block[
                int(0.05 * block.shape[1]) : int(0.95 * block.shape[1]),
                int(0.01 * block.shape[0]) : -int(0.01 * block.shape[0])
            ]
            cv2.imshow("", block)
            cv2.waitKey()
            columns = split_image_times(block, 1, 4)
            columns = columns[0]    
            print(len(columns))
            # for i in columns:
            #     cv2.imshow("", i)
            #     cv2.waitKey()
            # print(columns)

            proper_images = []
            for i, img in enumerate(columns):
                if img.size > 1500:
                    # cv2.imwrite(f"split/omr_column_block_{i}.jpg", img)
                    proper_images.append(img)

            data = []
            for k, _image in enumerate(proper_images):
                contours, _ = cv2.findContours(
                    _image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                inner_data = []
                i = 0
                for j, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    new_image = _image[y : y + h, x : x + w]
                    if 10000 > new_image.size > 1089:
                        cv2.imshow("", new_image)
                        cv2.waitKey()
                        inner_data.append(new_image)
                data.append(inner_data)
            print(len(data))
            # data = list(reversed(data))
            #####
            final = [[], [], [], [], []]
            for i in range(4):
                final[i].append(data[0][i])
                final[i].append(data[1][i])
                final[i].append(data[2][i])
                final[i].append(data[3][i])
                print(i)

            final = list(reversed(final))

            for i, img_list in enumerate(final):
                for j, img in enumerate(img_list):
                    cv2.imwrite(f"split/omr_{index}_{j + 1}.jpg", img)
                index += 1
            #######
            # data = [sorted_data_1, sorted_data_2, sorted_data_3, sorted_data_4, sorted_data_5]

            # for i, imga in enumerate(data):
            #     for j, img in enumerate(imga):
            #         cv2.imwrite(f"split/omr_{i}_{j}.jpg", img)

    # Now contour_image contains the first contour and all its child contours
    # cv2.imwrite(f"split/omr_area_area.jpg", image)

    cv2.destroyAllWindows()


# Usage example
image_path = "images/omr_sheet - filled.jpg"
read_omr(image_path)
