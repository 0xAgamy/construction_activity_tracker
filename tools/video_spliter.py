
import os
from pathlib import Path
import subprocess


def split_video(video_path, segment_duration=5):
   
    video_name = os.path.splitext(os.path.basename(video_path))[0]

   
    output_dir = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(output_dir, exist_ok=True)

    
    output_pattern = os.path.join(output_dir, f"{video_name}_%03d.mp4")

    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",                 
        "-map", "0",
        "-segment_time", str(segment_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]

    print(f"Splitting: {video_path}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Done: {video_name}")


def spliter(video_list, segment_duration=5):
    for video_path in video_list:
        if os.path.exists(video_path):
            split_video(video_path, segment_duration)
        else:
            print(f"File not found: {video_path}")


videos_path= Path("training_data/videos/")
videos= os.listdir(videos_path)
full_video_paths=[]
for video in videos:
    full_video_paths.append(videos_path / video)


spliter(list(full_video_paths), segment_duration=5)