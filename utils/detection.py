import torch
import numpy as np
from pathlib import Path

from model.yolov5.utils.general import set_logging,check_img_size,non_max_suppression,scale_coords
from model.yolov5.utils.torch_utils import select_device
from model.yolov5.models.experimental import attempt_load

class Detector:
    def __init__(self):
        self.weights = 'yolov5s.pt'
        self.imgsz = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.classes = [0, 24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73]
        self.augment = False
        self.agnostic_nms = False

        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        self.manager=None
        self.tracker=None

    def set_tracker(self, _tracker):
        self.tracker=_tracker

    def detect(self, _webcam, _path, _img, _im0s,_names):
        output=None
        _img = torch.from_numpy(_img).to(self.device)
        _img = _img.half() if self.half else _img.float()  # uint8 to fp16/32
        _img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if _img.ndimension() == 3:
            _img = _img.unsqueeze(0)

        # Inference
        pred = self.model(_img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        for i, det in enumerate(pred):  # detections per image
            if _webcam:  # batch_size >= 1
                p, s, im0 = Path(_path[i]), '%g: ' % i, _im0s[i].copy()
            else:
                p, s, im0 = Path(_path), '', _im0s

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(_img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh =[]
                confs=[]
                labels=[]
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append(conf)
                    labels.append(_names[int(cls)])

                self.tracker.update(im0, bbox_xywh, confs, labels)
            output = im0
        return output


def calculate_distance(_object, _person):
    obj_cen = np.array([_object.location[0].cpu() + _object.location[2].cpu()/2, _object.location[1].cpu() + _object.location[3].cpu()/2])
    per_cen = np.array([_person.location[0].cpu() + _person.location[2].cpu()/2, _person.location[1].cpu() + _person.location[3].cpu()/2])
    distance = np.sqrt(np.sum(np.square(obj_cen - per_cen)))
    return distance


def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
