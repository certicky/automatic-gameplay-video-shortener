# Required Libraries
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance

# Parameters for the trailer
trailer_duration = 45  # in seconds
splash_screen_duration = 5  # in seconds
game_name = "Shards of God"
game_url = "https://hvavra.itch.io/shards-of-god"
max_scene_duration = 10  # in seconds
crossfade_duration = 1  # in seconds, duration of the crossfade transition
video_path = "input.mp4"
output_path = "output.mp4"

# Load the video file
video_clip = VideoFileClip(video_path)

# Define threshold for scene change detection
threshold = np.mean([np.mean(frame) for frame in video_clip.iter_frames()]) * 1.1

# Initialize variables for scene detection
prev_frame = None
scene_changes = []
frame_count = 0

# Define the frame sampling rate (number of frames to skip between samples)
sample_rate = int(video_clip.fps)

# Scene detection
for t in range(int(video_clip.duration)):
    frame = video_clip.get_frame(t)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, frame_gray)
        if np.mean(frame_diff) > threshold:
            scene_changes.append(t)
    prev_frame = frame_gray

# Create a black background image with the same size as the video
background = Image.new('RGB', video_clip.size, 'black')

# Create a draw object
draw = ImageDraw.Draw(background)

# Define the font for the text
font = ImageFont.load_default()

# Get the width and height of the text
text = f"{game_name}\n{game_url}"
text_width, text_height = draw.textsize(text, font)

# Calculate the position for the text to be centered
x = (background.width - text_width) / 2
y = (background.height - text_height) / 2

# Draw the text on the background image
draw.text((x, y), text, fill='white', font=font)

# Create the initial splash screen clip
splash_screen = ImageClip(np.array(background), duration=splash_screen_duration)

# Initialize a list to store the clips for the trailer
trailer_clips = []

# Add the initial splash screen clip to the trailer clips
trailer_clips.append(splash_screen)

# Initialize the remaining trailer duration
remaining_duration = trailer_duration - 2 * splash_screen_duration

# Initialize a list to store the histograms of the scenes included in the trailer
included_histograms = []

# For each scene change
for t in scene_changes:
    if remaining_duration <= 0:
        break
    start_time = t
    end_time = min(t + max_scene_duration, video_clip.duration)
    scene_clip = video_clip.subclip(start_time, end_time)

    # Get the first frame of the scene and convert to 8-bit depth
    first_frame = scene_clip.get_frame(0) * 255
    first_frame = first_frame.astype(np.uint8)

    # Now we can use cvtColor with no error
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2HSV)    
    
    hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    if included_histograms:
        correlations = [distance.correlation(hist, included_hist) for included_hist in included_histograms]
        if max(correlations) > 0.5:
            continue
    included_histograms.append(hist)

    # Add a fade transition to the scene clip
    scene_clip = fadein(scene_clip, crossfade_duration)
    scene_clip = fadeout(scene_clip, crossfade_duration)
    
    trailer_clips.append(scene_clip)
    remaining_duration -= scene_clip.duration

# Create the final splash screen clip and add it to the trailer clips
trailer_clips.append(splash_screen)

# Concatenate the clips to form the final trailer
trailer = concatenate_videoclips(trailer_clips)

# Write the trailer to a file
trailer.write_videofile(output_path, codec='libx264', logger=None, verbose=False)

