import io

import cv2
import numpy as np
from botocore.handlers import disable_signing
import boto3
from tqdm.notebook import tqdm
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(bucket_name, bucket_prefix):
    estimator = BodyPoseEstimator(pretrained=True)

    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    bucket = resource.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=bucket_prefix+'/')

    for item in tqdm(objects):
        *_, file_name = item.key.split('/')
        img_stream = io.BytesIO()

        # Download image from S3
        obj = bucket.Object(item.key)
        obj.download_fileobj(img_stream)

        # Detect HumanPose on the image
        img_stream.seek(0)
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        keypoints = estimator(img)

        # draw keypoints and connections on the image
        img_output = draw_body_connections(img, keypoints, thickness=4, alpha=0.7)
        img_output = draw_keypoints(img_output, keypoints, radius=5, alpha=0.8)

        # Upload output image to S3
        img_bytes = cv2.imencode('.jpg', img_output)[1].tobytes()
        resource.Object(bucket_name, bucket_prefix+'_out/'+file_name).put(Body=img_bytes)


if __name__ == "__main__":
    BUCKET_NAME = 'l00157062'
    BUCKET_PREFIX = 'batch64'
    main(BUCKET_NAME, BUCKET_PREFIX)