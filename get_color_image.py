
import cv2
import numpy as np

# Load the image
img = cv2.imread('ball_high_n1.png')

# Select ROI
roi = cv2.selectROI(img)

# Crop image
roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# Convert to HSV
hsv_roi = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)

# Calculate average HSV values
average_color_per_row = np.average(hsv_roi, axis=0)
average_color = np.average(average_color_per_row, axis=0)

# Calculate min and max HSV values
min_color = np.min(hsv_roi, axis=(0, 1))
max_color = np.max(hsv_roi, axis=(0, 1))

print(f"Average HSV value in ROI: {average_color}")
print(f"Min HSV value in ROI: {min_color}")
print(f"Max HSV value in ROI: {max_color}")

cv2.waitKey(0)
cv2.destroyAllWindows()