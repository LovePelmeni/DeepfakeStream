import numpy 
import os 

def get_video_paths(source_path: str):
    paths = []
    for path in os.listdir(source_path):
        video_path = os.path.join(source_path, path)
        paths.append(video_path)
    return paths
