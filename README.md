# Automatic Game Trailer Creator

Automatic Game Trailer Creator is a simple Python script that generates a short video trailer from a full game playthrough. It uses scene detection to identify interesting moments, and then stitches these scenes together into a compact trailer.

## Requirements

- Python 3.7 or higher
- MoviePy (tested with 1.0.3)
- OpenCV
- NumPy
- SciPy
- Pillow
- librosa

You can install the required packages using pip:

```bash
pip install moviepy opencv-python numpy scipy pillow
```

## Usage
To create a trailer, run the script from the command line with three arguments: the name of the game, the subtitle, and the path to the input video.
For example:

```bash
python create_trailer.py "Shards of God" "Play for free at:\nhttps://hvavra.itch.io/shards-of-god" "input.mp4"
```

This will create a trailer for the game "Shards of God" using the video in "input.mp4", and output the resulting trailer as "output.mp4".