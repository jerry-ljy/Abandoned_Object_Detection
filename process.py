import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.yolov5.utils.datasets import LoadImages,LoadStreams
from utils.detection import Detector
from utils.tracking import Tracker
from numpy import random

import sys
sys.path.insert(0, './utils/yolov5')

def process(opt):
    source = opt.source
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    detector = Detector()
    imgsz=detector.imgsz
    model = detector.model
    device = detector.device
    half = detector.half

    tracker = Tracker()

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        output=detector.detect(webcam, path, img, im0s, names,tracker)
        cv2.imshow("result", output)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./jieyu.mp4', help='source')  # file/folder, 0 for webcam
    args = parser.parse_args()

    process(args)