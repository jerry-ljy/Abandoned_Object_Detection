import torch
import cv2
import os
from model.deep_sort.utils.parser import get_config
from model.deep_sort.deep_sort import DeepSort


class Tracker:
    def __init__(self):
        # initialize deepsort
        self.cfg = get_config()
        self.cfg.merge_from_file("./model/deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        self.manager = None

    def set_manager(self, _manager):
        self.manager = _manager
        self.deepsort.set_manager(_manager)


    def update(self, _im0, _bbox_xywh, _confs, _labels):
        xywhs = torch.Tensor(_bbox_xywh)
        confss = torch.Tensor(_confs)
        #
        # #####
        # # Pass detections to deepsort
        self.deepsort.update(xywhs, confss, _im0, _labels)

        self.manager.update()
        bbox_xyxy = []
        identities = []
        if self.manager.person_list or self.manager.object_list:
            for person in self.manager.person_list:
                bbox_xyxy.append(person.location)
                identities.append(person.id)
            for obj in self.manager.object_list:
                bbox_xyxy.append(obj.location)
                identities.append(obj.id)

        if bbox_xyxy and identities:
            draw_boxes(_im0, bbox_xyxy, identities, _manager=self.manager)



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def draw_boxes(img, bbox, identities=None, offset=(0, 0), _manager=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = tuple([0, 255, 0])
        name='person'
        if _manager.get_object(id):
            obj = _manager.get_object(id)
            name=obj.label
            if obj.is_abandoned:
                color = tuple([0, 0, 255])
        label = '{}{:d}{}'.format("", id, name)
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