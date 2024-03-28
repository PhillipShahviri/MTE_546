# import required libraries
import numpy as np
import cv2

frame_number = 500
cap = cv2.VideoCapture("ball-slo.mp4")
amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
res, frame = cap.read()
# print(frame.shape)
# print(amount_of_frames)
# print(frame)
imshow = cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# define a numpy.ndarray for the color
# here insert the bgr values which you want to convert to hsv
# green = np.uint8([[[0, 255, 0]]])
green = np.array(frame)

# convert the color to HSV
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
imshow = cv2.imshow("hsvGreen", hsvGreen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display the color values
# print("BGR of Green:", green)
# print("HSV of Green:", hsvGreen)
print("size of hsvGreen:", hsvGreen.shape)

# Compute the lower and upper limits
# lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
# upperLimit = hsvGreen[0][0][0] + 10, 255, 255
lowerLimit = hsvGreen[0][0][0]
upperLimit = hsvGreen[0][0][0]

# display the lower and upper limits
print("Lower Limit:",lowerLimit)
print("Upper Limit", upperLimit)