import cv2
import numpy as np
import math

TRANSF_SIZE = 600

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to filter tuples based on distance
def filter_tuples(points, max_distance=5):
    result = set()
    n = len(points)
    final_result = []
    for i in range(0, n):
        inter_results = set()
        for x in range(0, n):
            if calculate_distance(points[i], points[x]) < max_distance:
                inter_results.add(points[i])
                inter_results.add(points[x])
        result.add(tuple(inter_results))
    for i in result:
        final_result.append(i[0])
    return final_result

# Load target image and template
target_image = cv2.imread('images/omr_sheet.png')
template = cv2.imread('images/circle.png')

# Convert to grayscale
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Apply template matching
res = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Set a threshold
threshold = 0.8
loc = np.where(res >= threshold)

# Draw rectangles around matches
points = list(zip(*loc[::-1]))
points = filter_tuples(points, 5)

for pt in points:
    cv2.rectangle(target_image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 100, 0), 2)

points = sorted(points)
print(points)
x = [x[0] for x in points]
y = [y[1] for y in points]


pts1 = np.float32([(54, 312), (508, 312), (56, 724), (506, 724)])
pts2 = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(target_image, matrix, (600, 800))
result = result[template.shape[1] * 2:, :]
cv2.imwrite("images/test.png", result)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
