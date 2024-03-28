import cv2
import imutils

# Global variable to track if the mouse click has occurred
mouse_clicked = False
selected_point = None

START_Y_CROP = 230
END_Y_CROP = 686
START_X_CROP = 432
END_X_CROP = 1432

# Function to handle mouse events
def click_event(event, x, y, flags, param):
    global mouse_clicked, selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
        selected_point = (x, y)
        print("Selected Point: ({}, {})".format(x, y))

# Path to the video file
video_path = 'project_videos\\high\\Normal\\20231116_211200000_iOS.MOV'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame
ret, frame = cap.read()
#frame = frame[START_Y_CROP:END_Y_CROP, START_X_CROP:END_X_CROP]

#frame = imutils.resize(frame, width=600)

# Check if the frame is read successfully
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Display the first frame
cv2.imshow('Video Player', frame)

# Set the callback function for mouse events
cv2.setMouseCallback('Video Player', click_event)

# Wait for a mouse click
while not mouse_clicked:
    cv2.waitKey(1)

# Reset the mouse_clicked variable for video playback
mouse_clicked = False

# Play the video from the next frame until the end
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[START_Y_CROP:END_Y_CROP, START_X_CROP:END_X_CROP]
    if not ret:
        break

    # Display the frame
    cv2.imshow('Video Player', frame)

    # Break the loop if the user presses the 'Esc' key
    if cv2.waitKey(30) == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
