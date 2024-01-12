import cv2

def read_omr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image (e.g., convert to grayscale, apply thresholding, etc.)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform OMR processing (e.g., detect contours, analyze regions, etc.)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour and extract relevant information
    omr_area = None
    for i, contour in enumerate(contours):
        # Perform necessary operations on each contour (e.g., bounding rectangle, aspect ratio, etc.)
        x, y, w, h = cv2.boundingRect(contour)

        # Perform OMR logic based on the extracted information
        # cv2.imwrite(f"split/processed_image_{i}.jpg", image[y: y + h, x: x + w])
        
        if not omr_area:
            omr_area = (x,y, w, h)
        else:
            if w * h > omr_area[2] * omr_area[3]:
                omr_area = (x,y, w, h)
    
    x, y, w, h = omr_area
    omr_area = thresholded[y: y+h, x:x+w]
    
    
    # Display the processed image (optional)
    # cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
image_path = "images/omr_sheet.jpg"
read_omr(image_path)
