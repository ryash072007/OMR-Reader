import cv2
import numpy as np
import random

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mark_omr_sheet(image_path, output_path='graded_omr_sheet.jpg'):
    # Preprocess the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours = find_contours(thresh)

    # Initialize a dictionary to store question numbers and their corresponding answers
    answers = {}

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the threshold based on your OMR sheet
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresh[y:y + h, x:x + w]

            # Calculate the percentage of black pixels in the region
            percentage_filled = np.sum(roi == 0) / float(w * h)

            # Compare with a threshold to determine if the answer is marked
            if percentage_filled < 0.5:  # Adjust the threshold based on your sheet

                # Mark the region on the original image
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Save the marked image
    cv2.imwrite(output_path, img)
    print(f"Graded OMR sheet saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Assuming you have an answer key (1 for correct, 0 for incorrect)

    # Replace 'your_omr_sheet_image.jpg' with the path to your OMR sheet image
    omr_sheet_path = 'images/omr-filled.jpg'

    # Mark and grade the OMR sheet, save the result as 'graded_omr_sheet.jpg'
    mark_omr_sheet(omr_sheet_path, output_path='graded_omr_sheet.jpg')
