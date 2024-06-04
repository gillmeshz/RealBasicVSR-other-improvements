import cv2
import os

# 视频文件所在目录
video_directory = r'D:\anaconda3\envs\RealBasicVSR\RealBasicVSR\data'
# 获取目录中所有的mp4文件
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)

    # 创建以视频名命名的文件夹来保存图片
    output_folder = os.path.join(video_directory, os.path.splitext(video_file)[0])
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的原始帧率
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 计算每秒需要提取的帧数
    frames_per_second = 60

    # 计算跳帧间隔
    frame_skip = max(1, original_fps / frames_per_second)

    # 初始化帧计数器
    frame_count = 0

    #max_frames = 100 

    while True:
        ret, frame = cap.read()
        if not ret :  #or frame_count >= max_frames
            break

        if frame_count % frame_skip == 0:
        # 保存图片
            image_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(image_filename, frame)

        frame_count += 1

    # 释放视频对象
    cap.release()
  

