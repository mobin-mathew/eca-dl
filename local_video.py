import torch
import cv2

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def main(input_file_path, output_file_path):
    estimator = BodyPoseEstimator(pretrained=True)
    videoclip = cv2.VideoCapture(input_file_path)
    
    w = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_file_path, fourcc, fps, (w, h))

    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break
        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

        writer.write(frame)
    videoclip.release()


if __name__ == "__main__":
    input_file_path = "dance.mp4"
    output_file_path = "output.mp4"
    main(input_file_path, output_file_path)