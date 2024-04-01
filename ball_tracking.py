# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import imutils
import time
import csv
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


root_video_folder = 'project_videos'

# Initialize empty lists to store video files for each category
high_videos = {'Normal':[], 'Pocket':[], 'Vertical':[]}
medium_videos = {'Normal':[], 'Pocket':[], 'Vertical':[]}
low_videos = {'Normal':[], 'Pocket':[], 'Vertical':[]}

# list of tracked points
tracked_pts = []

# Ground truths
ground_truth = []


CROPPED_NET_Y_COORD = 350 # The y pixel coordinate of the cropped frame
ORIGINAL_NET_Y_COORD = 580
PIXELS_PER_METER = 894 # Points (546, 591) and (1291, 582) were selected on original image. x-diff is 745px / 3 ft => 894px / 1 m
RESIZED_IMG_PIXELS_PER_METER = 255.84 # Points (170, 185) and (404, 181) were selected on resized image. x-diff is 234px / 3 ft => 255.84px / 1 m
MAX_START_Y_COORD = 100 # If the y values are higher than this at the start of the tracked points then discard them

DIRECTION_INCOMING = 0
DIRECTION_OUTGOING = 1

SHOW_VIDEO = False
GET_DISTANCES = False
SHOW_PLOTS = False
PRINT_DATA = False
WRITE_DATA_TO_CSV = True
BALL_RADIUS_M = 0.04445
BALL_RADIUS_PX = BALL_RADIUS_M*PIXELS_PER_METER

START_Y_CROP = 230
END_Y_CROP = 686
START_X_CROP = 432
END_X_CROP = 1432


NUM_TESTS = 5


# Blue Ping Pong Ball HSV
ball_hsv_lower = (110, 127, 66)
ball_hsv_upper = (118, 255, 112)
# Min HSV value in ROI: [110 127  66]
# Max HSV value in ROI: [118 255 112]


# Masking tape HSV
masking_tape_hsv_lower = (19, 62, 201)
masking_tape_hsv_upper = (23, 81, 221)
# Min HSV value in ROI: [ 19  62 201]
# Max HSV value in ROI: [ 23  81 221]


# Black Tape HSV
black_tape_hsv_lower = (0, 0, 20)
black_tape_hsv_upper = (165, 54, 44)
# Min HSV value in ROI: [ 0  0 34]
# Max HSV value in ROI: [165  54  44]

# Initialize variables for mouseclicks
boundary_line_points = []
num_points = 0

# Function to handle mouse clicks
def mouse_callback(event, x, y, flags, params):

    global boundary_line_points, num_points
    # If the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if num_points < 2:
            boundary_line_points.append((x, y))
            num_points += 1
            
            


def get_video_files():
    num_files = 0
    for root, dirs, files in os.walk(root_video_folder):
        num_files += len(files)
        
    test_videos = {}
    ground_truth = {}
    
    # Iterate through the master folder
    for root, dirs, files in os.walk(root_video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.MOV', 'mov')) and file.startswith('Test'):
                test_num = file.split('Test')[1].split('_')[0].replace('-', '.')
                if "GroundTruth" in file: # If this is the ground truth, take note of the result
                    classification = file.split('.')[0].split('_')[-1]
                    if classification == "In":
                        ground_truth[test_num] = True
                    else:
                        ground_truth[test_num] = False
                else: # otherwise add video to list to be analyzed
                    if not test_num in test_videos.keys():
                        test_videos[test_num] = []
                    test_videos[test_num].append(os.path.join(root, file)) 
                
                
                
                
    return [test_videos, ground_truth]

def get_bounce_frame_index(tracked_pts):
    max_index = 0
    max_y = float('-inf')  # Initialize with negative infinity to ensure any valid value will be greater

    for index, item in enumerate(tracked_pts):
        if isinstance(item, tuple) and len(item) == 2:
            if item[1] > max_y:
                max_y = item[1]
                max_index = index
                
    return max_index

def get_ball_hsv(frame):
        # Select ROI
    roi = cv2.selectROI(frame)
    # Crop image
    roi_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    # Convert to HSV
    hsv_roi = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)

    # Calculate average HSV values
    average_color_per_row = np.average(hsv_roi, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    # Calculate min and max HSV values
    min_color = np.min(hsv_roi, axis=(0, 1))
    max_color = np.max(hsv_roi, axis=(0, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return min_color, max_color

def masking_tape_rect(frame, hsv_color_lower, hsv_color_upper):
    # Convert frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to isolate the desired color
    mask = cv2.inRange(hsv_frame, hsv_color_lower, hsv_color_upper)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find the largest rectangle
    largest_rectangle = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            largest_rectangle = (x, y, w, h)

    # Draw the largest rectangle on the original frame
    if largest_rectangle is not None:
        x, y, w, h = largest_rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.circle(frame, (int(x + w/2), int(y + h/2)), 5, (0, 0, 0))
    
    return (int(x + w/2), int(y + h/2))
    

def get_contour_centroid(contour):
    
    # Calculate the moments of the largest contour
    M = cv2.moments(contour)
    
    # Calculate centroid coordinates
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    
    return (centroid_x, centroid_y)


def get_midpoint(contour):
    # Calculate the average of all contour points
    contour_array = np.squeeze(contour)
    midpoint = np.mean(contour_array, axis=0, dtype=int)
    return tuple(midpoint)


def draw_boundary_line(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by arc length and get the four longest edges
    sorted_contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:6]
    
    # Calculate the center of the frame
    frame_center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Calculate the distances of midpoints from the center of the frame
    distances = [np.linalg.norm(np.array(get_midpoint(contour)) - np.array(frame_center)) for contour in sorted_contours]
    
    # Sort contours based on distances
    closest_contours_indices = np.argsort(distances)[:2]
    closest_contours = [sorted_contours[idx] for idx in closest_contours_indices]
    
    # Draw the two closest contours
    for contour in closest_contours:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    
    
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    
    
# Track the ball
def track_and_classify(video, mask_lower, mask_upper, video_name):
    global num_points, boundary_line_points
    max_y_frame_index = 0
    tracked_points = []

    while True:
        # grab the current frame
        frame = video.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if video else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        # crop the frame, blur it, and convert it to the HSV
        # color space
        
        #frame = frame[START_Y_CROP:END_Y_CROP, START_X_CROP:END_X_CROP]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "yellow", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, mask_lower, mask_upper)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=5)
        #cv2.imshow("mask", mask)
        #cv2.waitKey()

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((t, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 20:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(t), int(y)), int(radius),
                    (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # update the points queue
        tracked_points.append(center)

        # loop over the set of tracked points
        for i in range(1, len(tracked_points)):
            # if either of the tracked points are None, ignore them
            if tracked_points[i - 1] is None or tracked_points[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(512 / float(i + 1)) * 2.5)
            cv2.line(frame, tracked_points[i - 1], tracked_points[i], (0, 0, 255), thickness)
        
        if SHOW_VIDEO:
            # show the frame to our screen
            # if get_bounce_frame_index(tracked_points) > max_y_frame_index:
            #     print(str(get_bounce_frame_index(tracked_points)))
            #     max_y_frame_index = get_bounce_frame_index(tracked_points)
            cv2.imshow("Frame", frame)
                # cv2.waitKey()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
    
    
    
    max_y_frame_index = get_bounce_frame_index(tracked_points)
    bounce_point = tracked_points[max_y_frame_index]
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, max_y_frame_index)

    ret, frame = cap.read()
    
    
    
    draw_boundary_line(frame)
    out_centroid = masking_tape_rect(frame, masking_tape_hsv_lower, masking_tape_hsv_upper)
    
    cv2.namedWindow(video_name)
    cv2.setMouseCallback(video_name, mouse_callback)
    
    
    # Reset num_points and points
    num_points = 0
    del boundary_line_points[:]
    
    
    while True:
        # Draw the line if both points are selected
        if len(boundary_line_points) == 2:
            cv2.line(frame, boundary_line_points[0], boundary_line_points[1], (255, 0, 255), thickness=2)
            boundary_line_midpoint = (int((boundary_line_points[0][0] + boundary_line_points[1][0])/2), int((boundary_line_points[0][1] + boundary_line_points[1][1])/2))
            cv2.circle(frame, boundary_line_midpoint, 5, (0, 0, 0))
            
            
        # Display the resulting frame
        cv2.imshow(video_name, frame)

        # Wait for 'q' key to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    
    ball_to_out_dist_px = distance(bounce_point, out_centroid)
    line_to_out_dist_px = distance(boundary_line_midpoint, out_centroid)
    
    if (line_to_out_dist_px < ball_to_out_dist_px):
        print("BALL IN")
        return [True, ball_to_out_dist_px, line_to_out_dist_px]
    else:
        print("BALL OUT")
        return [False, ball_to_out_dist_px, line_to_out_dist_px]
    



def trim_nones_from_tuples(array):
    return [item for item in array if item is not None]



[test_videos, ground_truth] = get_video_files()

csv_name = "classification_data.csv"

headers = [["Test Number", "Filename", "Ball In", "Ball-Out Dist", "Line-Out Dist", "Ground Truth"]]

with open(csv_name, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(headers)

for test_num, videos in test_videos.items():
    for video in videos:
        if video:
            cap = cv2.VideoCapture(video)

            # Get the frame rate
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            print(f'Frame rate: {frame_rate} fps')
            
            [ball_in, ball_to_out_dist_px, line_to_out_dist_px] = track_and_classify(cap, ball_hsv_lower, ball_hsv_upper, video)
            
            
            tracked_pts = []
            
            cap.release()
            cv2.destroyAllWindows()
            
            if WRITE_DATA_TO_CSV:
                data = [[test_num, video, ball_in, ball_to_out_dist_px, line_to_out_dist_px, ground_truth[test_num]]]
                with open(csv_name, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                            

    
