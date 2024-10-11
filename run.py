# Required Libraries
import cv2, argparse, librosa, os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from scipy.spatial import distance
from sklearn.cluster import KMeans

# Create CLI arguments the parser
parser = argparse.ArgumentParser(description='Create a short video from longer gameplay footage.')

# Add the arguments
parser.add_argument('VIDEO_PATH', type=str, help='The path to the input video')
parser.add_argument('--trailer-duration', type=int, default=120, help='Optional: Approximate duration of the output video in seconds (default: 120)')
parser.add_argument('--max-scene-duration', type=int, default=7, help='Optional: Approximate duration of each clip (default: 7)')
parser.add_argument('--crossfade-duration', type=int, default=1, help='Optional: Duration of the cross-fade between clips (default: 1)')
parser.add_argument('--output-path', type=str, default='output.mp4', help='Optional: Path & name of the resulting file (default: output.mp4)')
parser.add_argument('--preserve-footage-ordering', type=bool, default=False, help='Optional: Preserve the ordering of clips from the input footage? (default: false)')

# Assign the arguments to variables
args = parser.parse_args()
VIDEO_PATH = args.VIDEO_PATH
TRAILER_DURATION = args.trailer_duration
MAX_SCENE_DURATION = args.max_scene_duration
CROSSFADE_DURATION = args.crossfade_duration
OUTPUT_PATH = args.output_path
PRESERVE_FOOTAGE_ORDERING = args.preserve_footage_ordering

def detect_scenes(video_clip, threshold, sample_rate=20):
    print("Detecting scenes...")
    prev_frame = None
    scene_changes = []
    for t in range(0, int(video_clip.duration), sample_rate):
        frame = video_clip.get_frame(t)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, frame_gray)
            if np.mean(frame_diff) > threshold:
                scene_changes.append(t)
        prev_frame = frame_gray
    print("Detected scenes:", len(scene_changes))
    return scene_changes

def merge_segments(segments):
    # Initialize merged segments with the first segment
    merged_segments = [segments[0]]
    
    for current_start, current_end in segments[1:]:
        # Get last segment in merged_segments list
        last_segment_start, last_segment_end = merged_segments[-1]
        
        # If the current segment overlaps with the last merged segment, merge them
        if current_start <= last_segment_end:
            merged_segments[-1] = (last_segment_start, max(last_segment_end, current_end))
        else:
            # If not, add it as a new segment
            merged_segments.append((current_start, current_end))
    
    return merged_segments

def get_loud_segments(scene_clip):
    FRAME_LENGTH_FACTOR = 0.05  # Frame size as a factor of sample rate
    AUDIO_PATH = "audio_temp.wav"  # Temp path to save audio file
    BUFFER = 1  # Buffer to add around each loud segment (in seconds)

    # Write audio to a wav file
    scene_clip.audio.write_audiofile(AUDIO_PATH, logger=None, verbose=False)

    # Load audio file with librosa
    y, sr = librosa.load(AUDIO_PATH)

    # Calculate the short time energy of the audio signal
    frame_length = int(sr * FRAME_LENGTH_FACTOR)
    hop_length = frame_length // 2
    energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])

    # Reshape energy for k-means
    energy = energy.reshape(-1, 1)

    # Use k-means to classify energy into two clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(energy)
    labels = kmeans.labels_

    # Identify cluster label for "loud" and "quiet"
    if np.mean(energy[labels == 0]) > np.mean(energy[labels == 1]):
        loud_cluster = 0
    else:
        loud_cluster = 1

    # Create segments
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:  # If the label changes, we have a new segment
            end = i
            if labels[i-1] == loud_cluster:  # If the previous segment was loud
                # Calculate segment start and end times (convert from frames to seconds)
                start_time = max(0, start * hop_length / sr - BUFFER)
                end_time = min(len(y) / sr, end * hop_length / sr + BUFFER)
                # Make sure segment duration is a multiple of 3 seconds
                duration = end_time - start_time
                rounded_duration = np.ceil(duration / 3) * 3
                end_time = start_time + rounded_duration
                # Add segment to list
                segments.append((start_time, end_time))
            start = i

    # Merge overlapping segments
    merged_segments = merge_segments(segments)

    return merged_segments

def create_trailer_clips(scene_changes, remaining_duration, max_scene_duration, visual_similarity_threshold = 0.5):
    print("Creating trailer clips...")
    # Keep track of visually similar groups of clips
    clip_groups = []
    group_histograms = []

    for t in scene_changes:
        if remaining_duration <= 0:
            break
        start_time = t
        end_time = min(t + max_scene_duration, video_clip.duration)
        scene_clip = video_clip.subclip(start_time, end_time)

        # get loud segments within the clip and move the start & end of a clip if there are some
        loud_segments = get_loud_segments(scene_clip)
        if len(loud_segments):
            new_start = None
            new_end = None
            for loud_start, loud_end in loud_segments:
                print(f"  Loud segment from {loud_start} to {loud_end} seconds.")
                if not new_start: new_start = loud_start
                new_end = loud_end
                if new_end > (new_start + max_scene_duration):
                    break
            start_time = t + new_start
            end_time = t + new_end
            print("  New clip duration:", end_time - start_time)
            scene_clip = video_clip.subclip(start_time, end_time)

        first_frame = scene_clip.get_frame(0) * 255
        first_frame = first_frame.astype(np.uint8)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2HSV)    

        # calculate histogram of the 1st frame
        hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # compare with existing groups of visually similar clips
        for i, group_hist in enumerate(group_histograms):
            if distance.correlation(hist, group_hist) > visual_similarity_threshold:
                # if it's visually similar to an existing group, add it to that group
                clip_groups[i].append({
                        "clip": scene_clip,
                        "start": start_time
                    })
                break
        else:
            # if it's not visually similar to any existing group, create a new group
            clip_groups.append([{
                "clip": scene_clip,
                "start": start_time
            }])
            group_histograms.append(hist)
    
    # Check if there are enough clip groups and if there aren't,
    # start again with higher threshold for visual similarity of clips
    if len(clip_groups) * MAX_SCENE_DURATION < TRAILER_DURATION:
        new_similarity_threshold = visual_similarity_threshold + ((1 - visual_similarity_threshold) / 2)
        if new_similarity_threshold <= 0.999:
            print("Too few groups of similar clips (" + str(len(clip_groups)) + ")! Starting again with similarity threshold of", new_similarity_threshold)
            return create_trailer_clips(scene_changes, remaining_duration, max_scene_duration, new_similarity_threshold)

    # Select the longest clip from each group
    trailer_clips = []
    print("There are", len(clip_groups), "groups of visually similar clips.")
    if not PRESERVE_FOOTAGE_ORDERING: clip_groups.sort(key=len, reverse=True)
    for group in clip_groups:
        longest_clip = group[0]["clip"]
        longest_clip_start_time = group[0]["start"]
        for gc in group:
            if gc["clip"].duration > longest_clip.duration:
                longest_clip = gc["clip"]
                longest_clip_start_time = gc["start"]

        print("Selecting a clip with duration of", longest_clip.duration, "from a group of", len(group), "visually similar clips.")
        if longest_clip.duration < max_scene_duration:
            print("  ... prolonging it to", max_scene_duration, "using start time of", longest_clip_start_time)
            longest_clip = video_clip.subclip(longest_clip_start_time, longest_clip_start_time + max_scene_duration)

        longest_clip = fadein(longest_clip, CROSSFADE_DURATION)
        longest_clip = fadeout(longest_clip, CROSSFADE_DURATION)
        longest_clip.audio = audio_fadein(longest_clip.audio, CROSSFADE_DURATION)
        longest_clip.audio = audio_fadeout(longest_clip.audio, CROSSFADE_DURATION)

        trailer_clips.append(longest_clip)
        remaining_duration -= longest_clip.duration
        if remaining_duration <= 0:
            break

    return trailer_clips


# Load the video file
print("Loading the video...")
video_clip = VideoFileClip(VIDEO_PATH).resize(height=720)  # Resize to meet Twitter's requirements
print("Loaded video FPS:", int(video_clip.fps))

# Calculate the threshold over a subset of frames
print("Computing threshold...")
num_samples = 250
interval = video_clip.duration // num_samples
sample_times = [i * interval for i in range(num_samples)]
frames = (video_clip.get_frame(t) for t in sample_times if t > 600) # calculate the threshold over uniformly chosen frames
threshold = np.mean([np.mean(frame) for frame in frames]) * 1.1

# Scene detection
scene_changes = detect_scenes(video_clip, threshold)

# Create trailer clips
remaining_duration = TRAILER_DURATION
trailer_clips = create_trailer_clips(scene_changes, remaining_duration, MAX_SCENE_DURATION)

# Concatenate the clips to form the final trailer
print("Concatenating the scenes into final video...")
trailer = concatenate_videoclips(trailer_clips)

# Write the trailer to a file
trailer.write_videofile(OUTPUT_PATH, codec='libx264', fps=30, bitrate="5000k", audio_codec='aac', audio_bitrate='128k', logger=None, verbose=False)
