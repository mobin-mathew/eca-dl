import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import cv2
from tqdm import tqdm

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def get_img_paths(img_folder):
    image_names = [x for x in os.listdir(img_folder) if x.endswith('.jpg') or x.endswith('.jpeg')]
    image_paths = [
        os.path.join(img_folder, image) 
        for image in image_names
        if image.endswith('.jpg') or image.endswith('.jpeg')
    ]
    return image_paths, image_names 


def load_img(path):
    return cv2.imread(path)


def save_img(img, path):
    cv2.imwrite(path, img)


def main(input_folder, output_folder, use_cuda=False):
    
    io_time = 0
    compute_time = 0
    other_time = 0

    model_download_time = time.time()
    estimator = BodyPoseEstimator(pretrained=True, use_cuda=use_cuda)
    model_download_time = time.time() - model_download_time

    other_t = time.time()
    image_paths, image_names = get_img_paths(input_folder)
    other_time += time.time() - other_t

    print(f'Processing {len(image_names)} images using {"GPU" if use_cuda else "CPU"} ...')
    for image_path, image_name in tqdm(zip(image_paths, image_names)):
        io_t = time.time()
        img_input = load_img(image_path)
        io_time += time.time() - io_t

        compute_t = time.time()
        keypoints = estimator(img_input)
        img_output = draw_body_connections(img_input, keypoints, thickness=4, alpha=0.7)
        img_output = draw_keypoints(img_output, keypoints, radius=5, alpha=0.8)
        compute_time += time.time() - compute_t

        io_t = time.time()
        save_path = os.path.join(output_folder, image_name)
        save_img(img_output, save_path)
        io_time += time.time() - io_t
    
    return other_time, io_time, compute_time


if __name__ == "__main__":
    input_folder = r"C:\Users\Mobin\Desktop\Mobin\eca-dl\data\image\batch32"
    use_cuda = False

    other_time = time.time()
    path, folder = os.path.split(input_folder)
    output_folder = os.path.join(path, folder+'_pose')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    other_ti = other_time + time.time()
    # print(f'-- other time is: {other_time} seconds')
    
    other_time, io_time, compute_time = main(input_folder, output_folder, use_cuda)
    print(f'-- other time is: {other_time+other_ti} seconds')
    print(f'-- io time is: {io_time} seconds')
    print(f'-- compute time is: {compute_time} seconds')