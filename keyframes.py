# 영상 파일로부터 key frames 추출(첫 프레임 포함)

import cv2
import os
import subprocess
from tqdm import tqdm
import pandas as pd

def extract_keyframes(video_path, base_folder, change_rate):
    # 비디오 파일명에서 확장자를 제외한 기본 이름 추출
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(base_folder, base_name)
    # 출력 디렉토리가 존재하지 않는 경우 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)
    # 첫 프레임 읽기
    success, prev_frame = video.read()  # prev_frame.shape=(356, 640, 3). 356(높이)x640(너비) 해상도의 컬러 이미지.
    if not success:
        raise Exception("비디오 파일을 읽는 데 실패했습니다.")
    # 첫 번째 프레임 포함
    keyframes = [prev_frame]
    frame_indices = [0]
    n_frames = 1

    while True:
        # 프레임 읽기
        success, curr_frame = video.read()
        if not success:
            break
        # 이전 프레임과 현재 프레임 사이의 절대 차이 계산
        diff = cv2.absdiff(curr_frame, prev_frame)
        if diff.sum() > prev_frame.sum() * change_rate:
            keyframes.append(curr_frame)
            frame_indices.append(n_frames)
        # 이전 프레임 업데이트
        prev_frame = curr_frame
        n_frames += 1

    # 키프레임을 특정 형식의 이름으로 저장
    for i, (k, idx) in enumerate(zip(keyframes, frame_indices)):
        frame_name = f"f_{i+1}_{idx}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), k)
    video.release()
    return n_frames, len(keyframes)



def extract_iframes(video_path, base_folder):
    # 비디오 파일 이름에서 확장자를 제외한 기본 이름 가져오기
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 기본 폴더 내에 비디오 이름을 기반으로 폴더 생성
    output_folder = os.path.join(base_folder, base_name)
    # 출력 폴더가 존재하지 않으면 새로 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ffprobe 명령어를 사용하여 총 프레임 수 가져오기
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path]
    total_frames = int(subprocess.check_output(ffprobe_cmd).decode().strip())

    # ffmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-i', video_path,  # 입력 비디오 파일
        '-vf', 'select=eq(pict_type\\,I)',  # I-프레임(키프레임)만 선택
        '-vsync', 'vfr',  # 가변 프레임 레이트로 출력
        '-f', 'image2',  # 출력 형식을 강제로 이미지 파일로 지정
        f'{output_folder}/f_%d.jpg'  # 출력 파일명 형식, %d는 프레임 번호
    ]
    # subprocess 모듈을 사용하여 ffmpeg 명령어 실행
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 추출된 I-프레임(키프레임) 수 계산
    i_frame_count = len([filename for filename in os.listdir(output_folder) if filename.endswith('.jpg')])

    return total_frames, i_frame_count
