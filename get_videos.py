import os

master_folder_path = 'Master_Folder'

# Initialize empty lists to store video files for each category
high_videos = []
medium_videos = []
low_videos = []

# Iterate through the master folder
for root, dirs, files in os.walk(master_folder_path):
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
                        high_videos.append(video_path)
                    elif category == 'medium':
                        medium_videos.append(video_path)
                    elif category == 'low':
                        low_videos.append(video_path)

# Print the lists of video files for each category
print("Videos in 'high':", high_videos)
print("Videos in 'medium':", medium_videos)
print("Videos in 'low':", low_videos)