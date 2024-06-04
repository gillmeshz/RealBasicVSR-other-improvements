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
