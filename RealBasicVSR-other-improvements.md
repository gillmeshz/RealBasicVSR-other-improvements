# RealBasicVSR-other-improvements

最近我写了一篇博客在CSDN上面：

[2024.5月RealBasicVSR超分辨率的使用详细教程和解决安装过程中遇到的问题](https://blog.csdn.net/m0_62557756/article/details/139393077)


在github上我想放一放我相关的一些测试和改善的代码.

## 测试

2024.6.3

我先从快手下载了一个视频，发现其分辨率如下：

![image-20240604210727326](C:\Users\Dell\AppData\Roaming\Typora\typora-user-images\image-20240604210727326.png)

其分辨率已经快达到警告的1000了，并且其帧率也有60，该视频是55s，通过视频分成图片的函数，最后得到了3000多张图片，所以运行RealBasicVSR给的源代码会出现超出显存的提示，基本上每次都会。

那我就想改良一下**inference_realbasicvsr.py**这个源代码。

<u>我的想法是这样的：</u>

1、因为图片非常的多，那么我一开始想将3000多张图片分成多个文件夹，再处理各个文件夹的图片，所以就写了一个代码：**split_images_into_folders.py**

2、但是我后面发现那和在一个文件夹分批次处理是一个道理

2、我测试了一下我的电脑显卡每次能处理几张图片，最后发现连10张都不行，那么我就分成了每次5张，刚好图片数也是5的倍数

3、给分批次处理加上一个进度条，可以查看其处理的进度

**效果如图所示：**

![运行](F:\阿里云盘Open\理工科学习\创建的虚拟环境\测试视频对比\运行.png)

发现处理了7个半小时才处理好，然后用代码将图片合成为视频，得到了我最终想要的。

**原图片占比：**

![1](F:\阿里云盘Open\理工科学习\创建的虚拟环境\测试视频对比\1.png)

**处理之后**

![2](F:\阿里云盘Open\理工科学习\创建的虚拟环境\测试视频对比\2.png)

## 代码

**inference_batch.py**

```python
import argparse
import glob
import os
import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img
from realbasicvsr.models.builder import build_model
from tqdm import tqdm

VIDEO_EXTENSIONS = ('.mp4', '.mov')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='maximum sequence length to be processed')
    parser.add_argument(
        '--is_save_as_png',
        type=bool,
        default=True,
        help='whether to save as png')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video')
    parser.add_argument(
        '--batch_size', type=int, default=100, help='Batch size for processing images')
    args = parser.parse_args()
    return args

def init_model(config, checkpoint=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config
    model.eval()
    return model

def process_batch(model, batch, device, output_dir, is_save_as_png, input_paths):
    with torch.no_grad():
        batch = batch.to(device)
        outputs = model(batch, test_mode=True)['output'].cpu()
        for i in range(outputs.size(1)):
            output = tensor2img(outputs[:, i, :, :, :])
            filename = os.path.basename(input_paths[i])
            if is_save_as_png:
                file_extension = os.path.splitext(filename)[1]
                filename = filename.replace(file_extension, '.png')
            mmcv.imwrite(output, f'{output_dir}/{filename}')

def main():
    args = parse_args()
    model = init_model(args.config, args.checkpoint)
    input_paths = sorted(glob.glob(f'{args.input_dir}/*'))
    os.makedirs(args.output_dir, exist_ok=True)
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True

    total_batches = (len(input_paths) + args.batch_size - 1) // args.batch_size
    with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
        for batch_start in range(0, len(input_paths), args.batch_size):
            batch_paths = input_paths[batch_start:batch_start + args.batch_size]
            inputs = []
            for input_path in batch_paths:
                img = mmcv.imread(input_path, channel_order='rgb')
                img = torch.from_numpy(img / 255.).permute(2, 0, 1).float().unsqueeze(0)
                inputs.append(img)
            inputs = torch.stack(inputs, dim=1)

            with torch.no_grad():
                if isinstance(args.max_seq_len, int):
                    outputs = []
                    for i in range(0, inputs.size(1), args.max_seq_len):
                        imgs = inputs[:, i:i + args.max_seq_len, :, :, :]
                        if cuda_flag:
                            imgs = imgs.cuda()
                        outputs.append(model(imgs, test_mode=True)['output'].cpu())
                    outputs = torch.cat(outputs, dim=1)
                else:
                    if cuda_flag:
                        inputs = inputs.cuda()
                    outputs = model(inputs, test_mode=True)['output'].cpu()

            if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
                output_dir = os.path.dirname(args.output_dir)
                mmcv.mkdir_or_exist(output_dir)

                h, w = outputs.shape[-2:]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output_dir, fourcc, args.fps,
                                               (w, h))
                for i in range(0, outputs.size(1)):
                    img = tensor2img(outputs[:, i, :, :, :])
                    video_writer.write(img.astype(np.uint8))
                cv2.destroyAllWindows()
                video_writer.release()
            else:
                mmcv.mkdir_or_exist(args.output_dir)
                for i in range(0, outputs.size(1)):
                    output = tensor2img(outputs[:, i, :, :, :])
                    filename = os.path.basename(batch_paths[i])
                    if args.is_save_as_png:
                        file_extension = os.path.splitext(filename)[1]
                        filename = filename.replace(file_extension, '.png')
                    mmcv.imwrite(output, f'{args.output_dir}/{filename}')

            pbar.update(1)

if __name__ == '__main__':
    main()

```

**video2picture.py**

```python
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
  


```

**image2video.py**

```python
import glob
import cv2
import mmcv
import numpy as np


class VideoDemo:
    ''' Generate video demo given a set of images.

    Args:
        input_dir (str): The directory storing the images.
        output_path (str): The path of the output video file.
        frame_rate (int): The frame rate of the output video.

    '''

    def __init__(self, input_dir, output_path, frame_rate):
        self.paths = sorted(glob.glob(f'{input_dir}/*'))
        self.output_path = output_path
        self.frame_rate = frame_rate

        # initialize video writer
        self.video_writer = None

    def __call__(self):
        for i, path in enumerate(self.paths):
            img = mmcv.imread(path, backend='cv2')

            # create video writer if haven't
            if self.video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc,
                                                    self.frame_rate, (w, h))

            self.video_writer.write(img.astype(np.uint8))

        cv2.destroyAllWindows()
        self.video_writer.release()


if __name__ == '__main__':
    """
    Assuming you have used our demo code to generate output images in
    results/demo_001. You can then use the following code to generate a video
    demo.
    """

    video_demo = VideoDemo(
        input_dir='G:/fuzi',
        output_path='G:/fuzi2/fuzi_video.mp4',
        frame_rate=60,
    )
    video_demo()

```

**split_images_into_folders.py**

```python
import os
import shutil
import argparse
from tqdm import tqdm

def split_images_into_folders(input_dir, output_base_dir, images_per_folder=100):
    # Ensure output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of all images in input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
    
    # Split images into multiple folders
    for i in tqdm(range(0, len(image_files), images_per_folder)):
        folder_name = f'folder_{i // images_per_folder + 1}'
        output_folder = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for image_file in image_files[i:i + images_per_folder]:
            shutil.copy(os.path.join(input_dir, image_file), output_folder)
    
    print(f"Images split into folders successfully.")

def parse_args():
    parser = argparse.ArgumentParser(description='Split images into folders.')
    parser.add_argument('input_dir', help='Directory of the input images')
    parser.add_argument('output_base_dir', help='Base directory for the output folders')
    parser.add_argument('--images_per_folder', type=int, default=100, help='Number of images per folder')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    split_images_into_folders(args.input_dir, args.output_base_dir, args.images_per_folder)

```

