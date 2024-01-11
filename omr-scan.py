import cv2
import numpy as np


def split_image_percent(image, h_percent, v_percent):
    
    # Calculate the split dimensions
    h_size = int(image.shape[0] * h_percent)
    v_size = int(image.shape[1] * v_percent)

    # Split the image
    images = split_image(image, h_size, v_size)

    return images


def split_image_times(image, h_times, v_times):
    
    # Calculate the split dimensions
    h_size = int(image.shape[0] / h_times)
    v_size = int(image.shape[1] / v_times)

    # Split the image
    images = split_image(image, h_size, v_size)
    
    return images


def split_image(image, h_size, v_size):
    images = []
    # Loop over the height of the image, stepping over 'h_size' pixels each time
    for i in range(0, image.shape[0], h_size):
        # Nested loop over the width of the image, stepping over 'v_size' pixels each time
        for j in range(0, image.shape[1], v_size):
            # Append a portion of the image of size 'h_size' x 'v_size' starting at (i, j) to the 'images' list
            images.append(image[i:i+h_size, j:j+v_size])

    return images


# Usage
if __name__ == "__main__":
    image = cv2.imread("images/omr-normal.jpeg")
    split = split_image_percent(image, 0.4, 1)
    for i, split_image_var in enumerate(split):
        cv2.imwrite(f"split{i}.jpeg", split_image_var)
    