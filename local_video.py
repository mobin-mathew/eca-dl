import time
import warnings
warnings.filterwarnings('ignore')

import torch
import cv2
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(input_file_path, output_file_path):
    io_time = 0
    compute_time = 0
    other_time = 0

    other_t = time.time()
    estimator = BodyPoseEstimator(pretrained=True, use_cuda=False)
    other_time += time.time() - other_t

    io_t = time.time() 
    videoclip = cv2.VideoCapture(input_file_path)
    
    w = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = int(videoclip.get(cv2.CAP_PROP_FOURCC))
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_file_path, fourcc, fps, (w, h))
    io_time += time.time() - io_t

    print('Processing Video..')
    def tqdm_generator():
        while videoclip.isOpened():
            yield
    
    for _ in tqdm(tqdm_generator()):
        io_t = time.time()
        flag, frame = videoclip.read()
        io_time += time.time() - io_t
        if not flag:
            break
        compute_t = time.time()
        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
        compute_time = time.time() - compute_t

        io_t = time.time()
        writer.write(frame)
        io_time = time.time() - io_t
    
    videoclip.release()

    return other_time, io_time, compute_time


if __name__ == "__main__":
    input_file_path = r"data\video\dance.mp4"
    output_file_path = r"data\video\output.mp4"
    other_time, io_time, compute_time = main(input_file_path, output_file_path)
    print(f'-- other time is: {other_time} seconds')
    print(f'-- io time is: {io_time} seconds')
    print(f'-- compute time is: {compute_time} seconds')