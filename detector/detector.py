import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import sys
sys.path.insert(0, '../yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords,set_logging
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.utils.plots import plot_one_box

class Detector:
    def __init__(self):
        self.weights = 'yolov5s.pt'
        self.imgsz = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.classes=None
        self.augment=False
        self.agnostic_nms=False

        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

    def detect(self, source):
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=self.imgsz)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path(path), '', im0s

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        location = [xyxy[i].cpu().numpy().tolist() for i in range(4)]
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # cv2.imshow(str(p), im0)
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                #     raise StopIteration
        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    detector = Detector()
    detector.detect('0')
