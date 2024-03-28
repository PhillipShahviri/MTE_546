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


CROPPED_NET_Y_COORD = 350 # The y pixel coordinate of the cropped frame
ORIGINAL_NET_Y_COORD = 580
PIXELS_PER_METER = 894 # Points (546, 591) and (1291, 582) were selected on original image. x-diff is 745px / 3 ft => 894px / 1 m
RESIZED_IMG_PIXELS_PER_METER = 255.84 # Points (170, 185) and (404, 181) were selected on resized image. x-diff is 234px / 3 ft => 255.84px / 1 m
MAX_START_Y_COORD = 100 # If the y values are higher than this at the start of the tracked points then discard them

DIRECTION_INCOMING = 0
DIRECTION_OUTGOING = 1

GET_DISTANCES = False
SHOW_PLOTS = True
PRINT_DATA = True
WRITE_DATA_TO_CSV = True
BALL_RADIUS_M = 0.04445
BALL_RADIUS_PX = BALL_RADIUS_M*PIXELS_PER_METER

START_Y_CROP = 230
END_Y_CROP = 686
START_X_CROP = 432
END_X_CROP = 1432
# Min HSV value in ROI: [20 42 91]
# Max HSV value in ROI: [ 30 255 248]
# define the lower and upper boundaries of the "yellow" ball in the HSV color space

# yellowLower = (20, 80, 100)
# yellowUpper = (30, 255, 248)


# TEST VALUES
yellowLower = (20, 125, 100)
yellowUpper = (30, 255, 248)

# Min HSV value in ROI: [ 21 214 202]
# Max HSV value in ROI: [ 25 255 245]


def get_video_files():
    # Iterate through the master folder
    for root, dirs, files in os.walk(root_video_folder):
        # Split the path into components
        path_components = root.split(os.path.sep)
        
        # Check if the path has enough components to identify category and subcategory
        if len(path_components) >= 3:
            _, category, subcategory = path_components[-3:]
            # Check if the current directory is a video subfolder
            if subcategory in ['Normal', 'Pocket', 'Vertical']:
                # Iterate through the files in the current subfolder
                for file in files:
                    # Check if the file is a video file (you may need to adjust this condition)
                    if file.endswith(('.mp4', '.avi', '.MOV')):
                        # Create the full path to the video file
                        video_path = os.path.join(root, file)
                        
                        # Append the video file to the appropriate list based on the category
                        if category == 'high':
                            high_videos[subcategory].append(video_path)
                        # elif category == 'medium':
                        #     medium_videos[subcategory].append(video_path)
                        # elif category == 'low':
                        #     low_videos[subcategory].append(video_path)

    print("Videos in 'high':", high_videos)
    print("Videos in 'medium':", medium_videos)
    print("Videos in 'low':", low_videos)


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

# Track the ball
def track_ball(video, tracked_points, mask_lower, mask_upper, show_video=True):
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
        
        frame = frame[START_Y_CROP:END_Y_CROP, START_X_CROP:END_X_CROP]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "yellow", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, mask_lower, mask_upper)
        mask = cv2.erode(mask, None, iterations=10)
        mask = cv2.dilate(mask, None, iterations=7)
        #cv2.imshow("mask", mask)
        
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
                    (0, 255, 255), 2)
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
        
        if show_video:
            # show the frame to our screen
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    return tracked_points


# Plot the trajectory
def interpolate_nones(A: np.ndarray):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A


def plot_trajectory(x1, y1, x2, y2, title1, title2):
    if x2 is None or y2 is None:
        plt.figure()
        plt.plot(x1, y1, label=title1)
        plt.gca().invert_yaxis()
        plt.xlabel('x-coordinate (pixels)')
        plt.ylabel('y-coordinate (pixels)')
        plt.legend()
        plt.show()
    else:
        plt.figure()
        plt.plot(x1, y1, label=title1)
        plt.plot(x2, y2, label=title2)
        plt.gca().invert_yaxis()
        plt.xlabel('x-coordinate (pixels)')
        plt.ylabel('y-coordinate (pixels)')
        plt.legend()
        plt.show()


def fit_line(x, y, order=1):
    coeffs = np.polyfit(x, y, order)
    fitted = np.polyval(coeffs, x)
    return fitted

def fit_parabola(x, y, order=2):
    coeffs = np.polyfit(x, y, order)
    fitted = np.polyval(coeffs, x)
    return fitted


def get_angle(x, y, direction):
    # Calculate the slope of the line between the first and last points
    if (direction == DIRECTION_INCOMING):
        slope = (y[-1] - y[-4]) / (x[-1] - x[-4]) # Get the slope from the points at the END of the incoming path
    elif (direction == DIRECTION_OUTGOING):
        slope = (y[3] - y[0]) / (x[3] - x[0]) # Get the sloe from the points at the START of the outgoing path

    # Calculate the angle in degrees
    angle = np.arctan(slope) * 180 / np.pi
    return angle

def plot_speeds(x, y, title1, title2):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.xlabel('Frame')
    plt.ylabel('Speed (m/s)')
    plt.title(title1)
    plt.subplot(2, 1, 2)
    plt.plot(y)
    plt.xlabel('Frame')
    plt.ylabel('Speed (m/s)')
    plt.title(title2)
    plt.tight_layout()
    plt.show()

def fit_curve(velocity_x, velocity_y, order):
    # cubic fit
    coeffs_vx = np.polyfit(np.linspace(0, len(velocity_x), len(velocity_x)), velocity_x, order)
    coeffs_vy = np.polyfit(np.linspace(0, len(velocity_y), len(velocity_y)), velocity_y, order)
    # Calculate the fitted speeds
    fitted_vx = np.polyval(coeffs_vx, np.linspace(0, len(velocity_x), len(velocity_x)))
    fitted_vy = np.polyval(coeffs_vy, np.linspace(0, len(velocity_y), len(velocity_y)))
    return fitted_vx, fitted_vy

# Mouse callback function
def click_event(event, x, y, flags, params, click_coordinates, frame):
    # If the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates.append((x, y))

        # Draw a circle where the user clicked
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # If two points have been clicked, draw a line between them
        if len(click_coordinates) == 2:
            cv2.line(frame, click_coordinates[0], click_coordinates[1], (0, 255, 0), 2)

        # Display the image
        cv2.imshow('image', frame)


# Calculate the distance between two points
def get_ball_distance_to_edge(frame, return_type='x'):
    # Display the image and set the mouse callback function
    points = []
    cv2.imshow('image', frame)
    cv2.setMouseCallback('image', lambda *args: click_event(*args, click_coordinates=points, frame=frame))

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If two points were clicked, calculate and print the distance between them
    if len(points) == 2:
        distance_to_edge = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        x_distance = points[1][0] - points[0][0]
        y_distance = points[1][1] - points[0][1]
        if return_type == 'x':
            return x_distance
        elif return_type == 'y':
            return y_distance
        else:
            return distance_to_edge
    else:
        return -1
        

def pad_arrays(arrays):
    # Find the maximum length among the arrays
    max_length = max(len(arr) for arr in arrays)
    
    # Pad each array with -1s to match the maximum length
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), constant_values=-1) for arr in arrays]
    
    return padded_arrays

def get_distance_in_frame(cap, frame_index):
    # Skip to the frame at max_index
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    except Exception as e:
        print(f"Error setting frame index: {e}")
    # Read the frame at max_index
    ret, frame = cap.read()
    if ret:
        frame = frame[START_Y_CROP:END_Y_CROP, START_X_CROP:END_X_CROP]

        # Initialize the list of points
        ball_x_distance = get_ball_distance_to_edge(frame)

        if (ball_x_distance != -1):
            ball_x_distance = np.round(ball_x_distance / PIXELS_PER_METER * 100, 2)

        return ball_x_distance
    else:
        return -1


def truncate_array_to_monotonic(arr, increment=1):
    result = [arr[0]]  # Initialize the result array with the first element

    for i in range(1, len(arr)):
        if arr[i] == result[-1] + increment:
            result.append(arr[i])
        elif arr[i] > result[-1] + increment:
            break  # Stop if the next value is greater than the expected increment

    return result

def trim_nones_from_tuples(array):
    non_none_indices = [i for i, value in enumerate(array) if (isinstance(value, tuple) and None not in value) or value is not None]

    if not non_none_indices:
        return []  # Handle case where all elements are None or tuples with None

    start_index = non_none_indices[0]
    
    while (array[start_index] is not None and array[start_index][1] > MAX_START_Y_COORD):
        start_index += 1

    end_index = non_none_indices[-1] + 1

    return array[start_index:end_index]


get_video_files()

video_files = {'high_videos': high_videos, 'medium_videos': medium_videos, 'low_videos': low_videos}

video = "project_videos\\high\\Normal\\20231116_211200000_iOS.MOV"


for video_type_name, video_type in video_files.items():
    for shot_type, videos in video_type.items():
        for video in videos:
            # video capture object
            cap = cv2.VideoCapture(video)

            # allow the camera or video file to warm up
            time.sleep(2.0)
            # Get the frame rate
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            print(f'Frame rate: {frame_rate} fps')


            tracked_pts = track_ball(cap, tracked_pts, yellowLower, yellowUpper)
               # print tracked_pts
            if PRINT_DATA:
                print(f'tracked_pts: {tracked_pts}')
    
            # cap.release()
            # cv2.destroyAllWindows()
            # print number of not None elements in tracked_pts
            print(f"Number of tracked points: {len([pt for pt in tracked_pts if pt is not None])}")

            trimmed_tracked_pts = trim_nones_from_tuples(tracked_pts)

            x_coords = interpolate_nones(np.array([pt[0] if pt is not None else np.nan for pt in tracked_pts]))
            y_coords = interpolate_nones(np.array([pt[1] if pt is not None else np.nan for pt in tracked_pts]))


            # print x_coords and y_coords
            if PRINT_DATA:
                print(f'x_coords: {x_coords}')
                print(f'y_coords: {y_coords}')

   
            # Find the indicies of frames where the ball is in contact with the net
            impact_indicies = (np.argwhere(y_coords > CROPPED_NET_Y_COORD - BALL_RADIUS_PX))
            impact_indicies = impact_indicies.flatten()

            # Truncate to remove indicies where the ball was not found in frame
            impact_indicies = truncate_array_to_monotonic(impact_indicies, increment=1) 
            impact_start_index = impact_indicies[0]
            print(f"Impact_start_index: {impact_start_index}")
            impact_end_index = impact_indicies[-1]
            print(f"Impact_end_index: {impact_end_index}")

            # Split x_coords and y_coords into incoming and outgoing arrays
            incoming_x = x_coords[:impact_start_index]
            incoming_y = y_coords[:impact_start_index]
            outgoing_x = x_coords[impact_end_index:]
            outgoing_y = y_coords[impact_end_index:]

            # print incoming_x, incoming_y, outgoing_x, and outgoing_y
            if PRINT_DATA:
                print(f'incoming_x: {incoming_x}')
                print(f'incoming_y: {incoming_y}')
                print(f'outgoing_x: {outgoing_x}')
                print(f'outgoing_y: {outgoing_y}')


            fitted_incoming_y = fit_parabola(incoming_x, incoming_y)
            fitted_outgoing_y = fit_parabola(outgoing_x, outgoing_y)

            # print fitted_incoming_y and fitted_outgoing_y
            if PRINT_DATA:
                print(f'fitted_incoming_y: {fitted_incoming_y}')
                print(f'fitted_outgoing_y: {fitted_outgoing_y}')



            incoming_angle = np.round(get_angle(incoming_x, fitted_incoming_y, direction=DIRECTION_INCOMING), 2)
            outgoing_angle = np.round(get_angle(outgoing_x, fitted_outgoing_y, direction=DIRECTION_OUTGOING), 2)

            if PRINT_DATA:
                print(f"angle of incoming trajectory: {incoming_angle} degrees")
                print(f"angle of outgoing trajectory: {outgoing_angle} degrees")

            # calculate Velocities
            incoming_vx = np.trim_zeros(np.diff(incoming_x) * frame_rate / PIXELS_PER_METER)
            incoming_vy = np.trim_zeros(np.diff(fitted_incoming_y) * frame_rate / PIXELS_PER_METER)
            outgoing_vx = np.trim_zeros(np.diff(outgoing_x) * frame_rate / PIXELS_PER_METER)
            outgoing_vy = np.trim_zeros(np.diff(fitted_outgoing_y) * frame_rate / PIXELS_PER_METER)

            # print incoming_vx, incoming_vy, outgoing_vx, and outgoing_vy
            if PRINT_DATA:
                print(f'incoming_vx: {incoming_vx}')
                print(f'incoming_vy: {incoming_vy}')
                print(f'outgoing_vx: {outgoing_vx}')
                print(f'outgoing_vy: {outgoing_vy}')

            fitted_incoming_vx, fitted_incoming_vy = fit_curve(incoming_vx, incoming_vy, 5)
            fitted_outgoing_vx, fitted_outgoing_vy = fit_curve(outgoing_vx, outgoing_vy, 3)

            # print fitted_incoming_vx, fitted_incoming_vy, fitted_outgoing_vx, and fitted_outgoing_vy
            if PRINT_DATA:
                print(f'fitted_incoming_vx: {fitted_incoming_vx}')
                print(f'fitted_incoming_vy: {fitted_incoming_vy}')
                print(f'fitted_outgoing_vx: {fitted_outgoing_vx}')
                print(f'fitted_outgoing_vy: {fitted_outgoing_vy}')


            median_incoming_vx = np.median(fitted_incoming_vx[-5:])
            median_incoming_vy = np.median(fitted_incoming_vy[-5:])
            
            median_outgoing_vx = np.median(fitted_outgoing_vx[:5])
            median_outgoing_vy = np.median(fitted_outgoing_vy[:5])


            # Release and reopen the video file before getting the distance from ball center to rim edge
            if GET_DISTANCES:
                cap.release()
                cap = cv2.VideoCapture(video)
                inbound_x_distance = get_distance_in_frame(cap, impact_start_index)
                print(f"Distance from ball to edge on impact: {inbound_x_distance} centimeters")
                cap.release()
                
                cap = cv2.VideoCapture(video)
                outbound_x_distance = get_distance_in_frame(cap, impact_end_index)
                print(f"Distance from ball to edge on exit: {outbound_x_distance} centimeters")
            else:
                inbound_x_distance = -1
                outbound_x_distance = -1

            cap.release()


            # print the fitted speeds
            if PRINT_DATA:
                print(f'Incoming fitted speed in x: {np.round(fitted_incoming_vx, 2)} m/s')

            if SHOW_PLOTS:
                # plot the trajectory
                plot_trajectory(x_coords, y_coords, None, None, 'Ball trajectory', None)
                # plot the incoming and outoging trajectories
                plot_trajectory(incoming_x, incoming_y, outgoing_x, outgoing_y, 'Incoming trajectory', 'Outgoing trajectory')
                # plot the incoming and outgoing fitted trajectories
                plot_trajectory(incoming_x, fitted_incoming_y, outgoing_x, fitted_outgoing_y, 'Incoming fit', 'Outgoing fit')
                # Plot the speeds in x and y as separate sub plots
                plot_speeds(incoming_vx, incoming_vy, 'Incoming speed in x', 'Incoming speed in y')
                plot_speeds(outgoing_vx, outgoing_vy, 'Outgoing speed in x', 'Outgoing speed in y')
                # Plot the fitted speeds in x and y as separate sub plots
                plot_speeds(fitted_incoming_vx, fitted_incoming_vy, 'Incoming fitted speed in x', 'Incoming fitted speed in y')
                plot_speeds(fitted_outgoing_vx, fitted_outgoing_vy, 'Outgoing fitted speed in x', 'Outgoing fitted speed in y')




            if WRITE_DATA_TO_CSV:
                #data_arrays = [median_incoming_vx, median_incoming_vy, median_outgoing_vx, median_outgoing_vy, [inbound_x_distance], [outbound_x_distance], [incoming_angle], [outgoing_angle]]
                #padded_arrays = pad_arrays(data_arrays)
                
                # Assuming fitted_incoming_vx, fitted_incoming_vy, ball_x_distance, incoming_angle, and outgoing_angle are defined
                # ball_data = {
                # 	'median_incoming_vx': padded_arrays[0],
                # 	'median_incoming_vy': padded_arrays[1],
                # 	"median_outgoing_vx": padded_arrays[2],
                # 	"median_outgoing_vy": padded_arrays[3],
                # 	'inbound_x_distance': padded_arrays[4],
                # 	'outbound_x_distance': padded_arrays[5],
                # 	'incoming_angle': padded_arrays[6],
                # 	'outgoing_angle': padded_arrays[7]
                # }
                
                ball_data = {
                    'median_incoming_vx': [median_incoming_vx],
                    'median_incoming_vy': [median_incoming_vy],
                    "median_outgoing_vx": [median_outgoing_vx],
                    "median_outgoing_vy": [median_outgoing_vy],
                    'inbound_x_distance': [inbound_x_distance],
                    'outbound_x_distance': [outbound_x_distance],
                    'incoming_angle': [incoming_angle],
                    'outgoing_angle': [outgoing_angle]
                }
                
                
                #print("Hello {} {}, hope you're well!".format(first_name,last_name))

                # Create a DataFrame from the data
                video_name_df = pd.DataFrame({"{} {}".format(video_type_name, shot_type): [video]})
                ball_data_df = pd.DataFrame(ball_data)
                empty_df = pd.DataFrame(columns=range(8))

                # Write the DataFrame to a CSV file
                video_name_df.to_csv(video_type_name + '.csv', index=False, mode='a')
                ball_data_df.to_csv(video_type_name + '.csv', index=False, mode='a')
                empty_df.loc[0] = [None] * 8
                empty_df.to_csv(video_type_name + '.csv', mode='a', header=False, index=False)
                empty_df.to_csv(video_type_name + '.csv', mode='a', header=False, index=False)
                empty_df.to_csv(video_type_name + '.csv', mode='a', header=False, index=False)

                tracked_pts = []
