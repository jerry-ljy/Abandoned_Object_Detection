import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.datasets import LoadImages,LoadStreams

from .detection import Detector
from .tracking import Tracker
from .manager import Manager
from numpy import random

import sys
sys.path.insert(0, './model/yolov5')

class ABDetector:
    def __init__(self):
        self.detector = Detector()
        self.manager = Manager()

        self.tracker = Tracker()
        self.tracker.set_manager(self.manager)

        self.detector.set_tracker(self.tracker)

    def process(self, opt):
        output_path = opt.output
        source = opt.source
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        imgsz = self.detector.imgsz
        model = self.detector.model
        device = self.detector.device
        half = self.detector.half

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

        vid_path, vid_writer = None, None
        for path, img, im0s, vid_cap in dataset:
            output = self.detector.detect(webcam, path, img, im0s, names)
            cv2.imshow("result", output)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            if (not webcam) and (output!=''):
                if vid_path != output_path:
                    vid_path = output_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(output)
