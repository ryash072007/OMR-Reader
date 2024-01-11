import cv2
import numpy as np

def find_all_instances_rotated_and_flipped(target_image_path, source_image_path, threshold=0.9, rotation_range=360, rotation_step=10):
    # Read the images
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)

    # Extract the width and height of the target image
    target_width, target_height = target_image.shape[::-1]

    # Iterate over rotation angles
    for angle in range(0, rotation_range, rotation_step):
        rotated_target_image = rotate_image(target_image, angle)

        # Check both original and flipped versions
        for flip in [False, True]:
            if flip:
                rotated_target_image = cv2.flip(rotated_target_image, 1)  # 1 means horizontal flip

            # Use the TM_CCOEFF_NORMED method for template matching
            result = cv2.matchTemplate(source_image, rotated_target_image, cv2.TM_CCOEFF_NORMED)
            
            # Find the locations where the correlation coefficient is above the threshold
            locations = np.where(result >= threshold)
            # Draw rectangles around all matches
            for loc in zip(*locations[::-1]):
                top_left = loc
                bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
                print((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)
                cv2.rectangle(source_image, top_left, bottom_right, 100, 2)

    # Display the result
    cv2.imshow('Matching Result', source_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_image(image, angle):
    # Rotate the image by the specified angle
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Example usage
target_image_path = 'images/corner.png'
source_image_path = 'images/omr-filled.jpg'

find_all_instances_rotated_and_flipped(target_image_path, source_image_path)
