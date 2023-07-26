# Required Libraries
import cv2, argparse
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance

# Config Parameters
TRAILER_DURATION = 45
SPLASH_SCREEN_DURATION = 5
MAX_SCENE_DURATION = 10
CROSSFADE_DURATION = 1
OUTPUT_PATH = "output.mp4"
FONT_SIZE = 70
FONT_SIZE_SUBTITLE = 18
BLUR_RADIUS = 47

# Create the parser
parser = argparse.ArgumentParser(description='Create a game trailer from gameplay footage.')

# Add the arguments
parser.add_argument('GAME_NAME', type=str, help='The name of the game')
parser.add_argument('SUB_TITLE', type=str, help='The subtitle text')
parser.add_argument('VIDEO_PATH', type=str, help='The path to the input video')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
GAME_NAME = args.GAME_NAME
SUB_TITLE = args.SUB_TITLE
VIDEO_PATH = args.VIDEO_PATH

def detect_scenes(video_clip, threshold):
    prev_frame = None
    scene_changes = []
    for t in range(int(video_clip.duration)):
        frame = video_clip.get_frame(t)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, frame_gray)
            if np.mean(frame_diff) > threshold:
                scene_changes.append(t)
        prev_frame = frame_gray
    return scene_changes


def create_blurred_clip(middle_scene_clip, blur_radius):
    blurred_scene_frames = [cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0) 
                            for frame in middle_scene_clip.iter_frames()]
    blurred_scene_clip = ImageSequenceClip(blurred_scene_frames, fps=middle_scene_clip.fps)
    return blurred_scene_clip

def create_text_clip(text, font, duration, pos):
    text_image = Image.new('RGBA', video_clip.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    text_width, text_height = draw.textsize(text, font)
    text_x = (text_image.width - text_width) / 2
    text_y = (text_image.height - text_height) / 2
    draw.text((text_x, text_y), text, fill='white', font=font)
    text_clip = ImageClip(np.array(text_image)).set_duration(duration)
    if isinstance(pos, tuple) and len(pos) == 2 and pos[0] == "center":
        text_clip = text_clip.set_position(lambda t: ("center", pos[1] + t * 0))
    else:
        text_clip = text_clip.set_position(pos)
    return text_clip

def create_trailer_clips(scene_changes, remaining_duration, max_scene_duration):
    trailer_clips = []
    included_histograms = []
    for t in scene_changes:
        if remaining_duration <= 0:
            break
        start_time = t
        end_time = min(t + max_scene_duration, video_clip.duration)
        scene_clip = video_clip.subclip(start_time, end_time)

        first_frame = scene_clip.get_frame(0) * 255
        first_frame = first_frame.astype(np.uint8)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2HSV)    

        hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if included_histograms:
            correlations = [distance.correlation(hist, included_hist) for included_hist in included_histograms]
            if max(correlations) > 0.5:
                continue
        included_histograms.append(hist)

        scene_clip = fadein(scene_clip, CROSSFADE_DURATION)
        scene_clip = fadeout(scene_clip, CROSSFADE_DURATION)
        trailer_clips.append(scene_clip)
        remaining_duration -= scene_clip.duration
    return trailer_clips


# Load the video file
video_clip = VideoFileClip(VIDEO_PATH)

# Scene detection
threshold = np.mean([np.mean(frame) for frame in video_clip.iter_frames()]) * 1.1
scene_changes = detect_scenes(video_clip, threshold)

# Create intro and outro
middle_scene_time = scene_changes[len(scene_changes)//2]
middle_scene_clip = video_clip.subclip(middle_scene_time, min(middle_scene_time + SPLASH_SCREEN_DURATION, video_clip.duration))
blurred_scene_clip = create_blurred_clip(middle_scene_clip, BLUR_RADIUS)
font_intro = ImageFont.truetype("arial_bold.ttf", FONT_SIZE)
intro = CompositeVideoClip([blurred_scene_clip, create_text_clip(GAME_NAME, font_intro, SPLASH_SCREEN_DURATION, "center")])

font_outro_title = ImageFont.truetype("arial_bold.ttf", int(FONT_SIZE / 2))
font_outro_url = ImageFont.truetype("arial_bold.ttf", FONT_SIZE_SUBTITLE)

title_clip = create_text_clip(GAME_NAME, font_outro_title, SPLASH_SCREEN_DURATION, "center")
url_clip = create_text_clip(SUB_TITLE, font_outro_url, SPLASH_SCREEN_DURATION, ("center", int(FONT_SIZE / 2) + 25))
outro = CompositeVideoClip([title_clip, url_clip])

# Create trailer clips
remaining_duration = TRAILER_DURATION - 2 * SPLASH_SCREEN_DURATION
trailer_clips = create_trailer_clips(scene_changes, remaining_duration, MAX_SCENE_DURATION)

# Add intro and outro to the trailer
trailer_clips.insert(0, intro)
trailer_clips.append(outro)

# Concatenate the clips to form the final trailer
trailer = concatenate_videoclips(trailer_clips)

# Write the trailer to a file
trailer.write_videofile(OUTPUT_PATH, codec='libx264', logger=None, verbose=False)
