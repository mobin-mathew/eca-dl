import io

import cv2
import numpy as np
from botocore.handlers import disable_signing
import boto3
from tqdm.notebook import tqdm
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(bucket_name, file_name):
    estimator = BodyPoseEstimator(pretrained=True)

    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    # download video from s3
    bucket = resource.Bucket(bucket_name)
    obj = bucket.Object(file_name)
    obj.download_file('temp.mp4')

    videoclip = cv2.VideoCapture('temp.mp4')

    w = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = int(videoclip.get(cv2.CAP_PROP_FOURCC))
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('temp_out.mp4', fourcc, fps, (w, h))

    while videoclip.isOpened():
        hasframe, frame = videoclip.read()
        if not hasframe:
            break

        # get keypoints on frame using deep learning model
        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
        writer.write(frame)
    videoclip.release()

    with open('temp_out.mp4', 'rb') as file:
        bucket.upload_fileobj(file, 'output_'+file_name)

    # bucket.upload_file('temp_out.mp4', 'output_'+file_name)


if __name__ == "__main__":
    BUCKET_NAME = 'l00157062'
    FILE_NAME = 'dance.mp4'
    main(BUCKET_NAME, FILE_NAME)