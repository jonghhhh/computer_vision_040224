# 영상 파일로부터 key frames 추출

import cv2           #pip install opencv-python
import os
from tqdm import tqdm
import pandas as pd

def extract_key_frames(video_path, output_path, change_rate):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Read the first frame
    success, prev_frame = video.read()  # prev_frame.shape=(356, 640, 3). 356(높이)x640(너비) 해상도의 컬러 이미지.
    if not success:
        raise Exception("Failed to read the video file.")
    n_frames, keyframes=1, []
    while True:
        # Read frames
        success, curr_frame = video.read()
        if not success:
            break
        # Calculate the absolute difference between the frames
        diff = cv2.absdiff(curr_frame, prev_frame) #프레임 비교
        if diff.sum() > prev_frame.sum()*change_rate:
            keyframes.append([n_frames, curr_frame])
        else:
            pass
        # Update the previous frame
        prev_frame = curr_frame
        n_frames += 1
    for i, [n_frames, keyframe] in enumerate(keyframes):
        frame_name = f"f{str(n_frames)}_{str(i+1)}.jpg"
        cv2.imwrite(f"{output_path}/{frame_name}", keyframe)
    video.release()
    return n_frames, len(keyframes) # 총 프레임 개수, cutoff 해당하는 diff.sum()
