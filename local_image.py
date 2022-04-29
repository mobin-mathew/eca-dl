import os

import torch
import cv2

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def get_img_paths(img_folder):
    image_names = os.listdir(img_folder)
    image_paths = [
        os.path.join(img_folder, image) 
        for image in image_names
        if image.endswith('.jpg') or image.endswith('.jpeg')
    ]
    return image_paths, image_names 


def load_img(path):
    return cv2.imread(path)


def save_img(img, path):
    print(img, path)
    cv2.imwrite(path, img)


def main(input_folder, output_folder):
    estimator = BodyPoseEstimator(pretrained=True)

    for image_path, image_name in zip(*get_img_paths(input_folder)):
        img_input = load_img(image_path)

        keypoints = estimator(img_input)
        img_output = draw_body_connections(img_input, keypoints, thickness=4, alpha=0.7)
        img_output = draw_keypoints(img_output, keypoints, radius=5, alpha=0.8)

        save_path = os.path.join(output_folder, image_name)
        save_img(img_output, save_path)


if __name__ == "__main__":
    input_folder = r"C:\Users\Mobin\Desktop\hardik\batch64"
    path, folder = os.path.split(input_folder)
    output_folder = os.path.join(path, folder+'_pose')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    main(input_folder, output_folder)