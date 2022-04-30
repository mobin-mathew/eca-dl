import io
import time
import warnings
warnings.filterwarnings('ignore')


import cv2
import numpy as np
from botocore.handlers import disable_signing
import boto3
from tqdm.notebook import tqdm
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(bucket_name, bucket_prefix):

    io_time = 0
    compute_time = 0
    other_time = 0

    model_download_time = time.time()
    estimator = BodyPoseEstimator(pretrained=True)
    model_download_time = time.time() - model_download_time

    other_t = time.time()
    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    bucket = resource.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=bucket_prefix+'/')
    other_time = time.time() - other_t

    print('Processing Images: ')
    for item in tqdm(objects):
        
        other_t = time.time()
        *_, file_name = item.key.split('/')
        img_stream = io.BytesIO()
        other_time  += time.time() - other_t

        # Download image from S3
        io_t = time.time()
        obj = bucket.Object(item.key)
        obj.download_fileobj(img_stream)
        io_time += time.time() - other_t

        comput_t = time.time()
        img_stream.seek(0)
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Detect HumanPose on the image
        keypoints = estimator(img)

        # draw keypoints and connections on the image
        img_output = draw_body_connections(img, keypoints, thickness=4, alpha=0.7)
        img_output = draw_keypoints(img_output, keypoints, radius=5, alpha=0.8)

        compute_time += time.time() - comput_t

        # Upload output image to S3
        io_t = time.time()
        img_bytes = cv2.imencode('.jpg', img_output)[1].tobytes()
        resource.Object(bucket_name, bucket_prefix+'_out/'+file_name).put(Body=img_bytes)
        io_time += time.time() - io_t

    return other_time, io_time, compute_time
    
if __name__ == "__main__":
    BUCKET_NAME = 'l00157062'
    BUCKET_PREFIX = 'batch64'
    other_time, io_time, compute_time = main(BUCKET_NAME, BUCKET_PREFIX)
    print(f'-- other time is: {other_time} seconds')
    print(f'-- io time is: {io_time} seconds')
    print(f'-- compute time is: {compute_time} seconds')