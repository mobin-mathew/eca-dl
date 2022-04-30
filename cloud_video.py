import io
import time

import cv2
import numpy as np
from botocore.handlers import disable_signing
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm.notebook import tqdm
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(bucket_name, file_name):

    io_time = 0
    compute_time = 0
    other_time = 0

    other_t = time.time()
    estimator = BodyPoseEstimator(pretrained=True)

    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    other_time += time.time() - other_t

    io_t = time.time()
    # download video from s3
    bucket = resource.Bucket(bucket_name)
    obj = bucket.Object(file_name)
    print('Downloading file from S3..')
    obj.download_file('temp.mp4')

    videoclip = cv2.VideoCapture('temp.mp4')

    w = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = int(videoclip.get(cv2.CAP_PROP_FOURCC))
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('temp_out.mp4', fourcc, fps, (w, h))
    io_time += time.time() - io_t

    print('Processing Video..')
    def tqdm_generator():
        while videoclip.isOpened():
            yield

    for _ in tqdm(tqdm_generator()):
        io_t = time.time()
        hasframe, frame = videoclip.read()
        io_time += time.time() - io_t
        if not hasframe:
            break

        # get keypoints on frame using deep learning model
        comput_t = time.time()
        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
        compute_time += time.time() - comput_t

        io_t = time.time()
        writer.write(frame)
        io_time += time.time() - io_t

    videoclip.release()

    print('Uploading output vido to s3..')
    io_t = time.time()
    transfer_config = TransferConfig(multipart_threshold=28388608)
    with open('temp_out.mp4', 'rb') as file:
        bucket.upload_fileobj(file, 'output_'+file_name, Config=transfer_config)
    io_time = time.time() - io_t

    # bucket.upload_file('temp_out.mp4', 'output_'+file_name)
    return other_time, io_time, compute_time


if __name__ == "__main__":
    BUCKET_NAME = 'l00157062'
    FILE_NAME = 'dance.mp4'
    other_time, io_time, compute_time = main(BUCKET_NAME, FILE_NAME)
    print(f'-- other time is: {other_time} seconds')
    print(f'-- io time is: {io_time} seconds')
    print(f'-- compute time is: {compute_time} seconds')