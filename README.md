# Automatic Gameplay Video Shortener

Automatic Gameplay Video Shortener is a simple Python script that generates a short video from a full game playthrough. It uses scene detection to identify interesting moments, and then stitches these scenes together into a compact trailer.

Why? You can upload the resulting short gameplay video to Twitter, where there's a 2-minute limit on video length.

## Requirements

- Python 3.7 or higher
- MoviePy (tested with 1.0.3)
- OpenCV
- NumPy
- SciPy
- librosa

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
To create a short video, run the script from the command line with the path to the input long video.
For example:

```bash
python run.py "input.mp4"
```

This will shorten the video in "input.mp4", and output the result as "output.mp4".
