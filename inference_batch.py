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
