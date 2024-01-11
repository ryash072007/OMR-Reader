import cv2
import numpy as np

def find_image_in_another(target_image_path, source_image_path):
    # Read the images
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)

    # Use the TM_CCOEFF_NORMED method for template matching
    method = cv2.TM_CCOEFF_NORMED

    # Apply template matching
    result = cv2.matchTemplate(source_image, target_image, method)
    
    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Extract the width and height of the target image
    target_width, target_height = target_image.shape[::-1]

    # Draw a rectangle around the matched region
    top_left = max_loc
    bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
    cv2.rectangle(source_image, top_left, bottom_right, 100, 2)

    # Display the result
    cv2.imshow('Matching Result', source_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
target_image_path = 'images/corner.png'
source_image_path = 'images/omr-filled.jpg'

find_image_in_another(target_image_path, source_image_path)
