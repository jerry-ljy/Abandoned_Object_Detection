import torch
import cv2
from pathlib import Path

from utils.deep_sort.utils.parser import get_config
from utils.deep_sort.deep_sort import DeepSort
from utils.yolov5.utils.general import scale_coords

class Tracker:
    def __init__(self):
        # initialize deepsort
        self.cfg = get_config()
        self.cfg.merge_from_file("./utils/deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def update(self, _object_list, _im0):
        bbox_xywh = []
        confs = []

        for object in _object_list:
            bbox_xywh.append(object.location)
            confs.append(object.confs)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)
        #
        # #####
        # # Pass detections to deepsort
        outputs = self.deepsort.update(xywhs, confss, _im0)
        #
        # # draw boxes for visualization
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes(_im0, bbox_xyxy, identities)


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = [int((p * (id ** 2 - id + 1)) % 255) for p in palette]
        color = tuple(color)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

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