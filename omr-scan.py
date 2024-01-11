import cv2
import numpy as np


def split_image_percent(image, h_percent):
    # Calculate the split dimensions
    h_size = int(image.shape[0] * h_percent)

    # Split the image
    images = (image[:h_size, :], image[h_size:, :])

    return images


def split_image_percent_v(image, v_percent):
    # Calculate the split dimensions
    v_size = int(image.shape[1] * v_percent)

    # Split the image
    images = (image[:, v_size:], image[:, :v_size])

    return images


def split_image_times(image, h_times, v_times):
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)

    # Split the image
    images = []
    for i in range(0, image.shape[0], h_size):
        for j in range(0, image.shape[1], v_size):
            images.append(image[i : i + h_size, j : j + v_size])

    return images


def split_image(image, h_size, v_size):
    images = []
    # Loop over the height of the image, stepping over 'h_size' pixels each time
    for i in range(0, image.shape[0], h_size):
        # Nested loop over the width of the image, stepping over 'v_size' pixels each time
        for j in range(0, image.shape[1], v_size):
            # Append a portion of the image of size 'h_size' x 'v_size' starting at (i, j) to the 'images' list
            images.append(image[i : i + h_size, j : j + v_size])

    return images


# Usage
if __name__ == "__main__":
    image = cv2.imread("images/omr-filled.jpg")
    _, bottom = split_image_percent(image, 0.44518)
    up, _ = split_image_percent(bottom, 0.94698)
    cv2.imshow("", up)
    cv2.waitKey()
    
    # _, right = split_image_percent_v(up, 0.95)
    # left, _ = split_image_percent_v(right, 0.07)
    # cv2.imwrite(f"left.jpg", left)
    # split_images = split_image_times(left, 25, 4)
    # for i, img in enumerate(split_images):
    #     cv2.imwrite(f"split/split_image_{i}.jpg", img)
