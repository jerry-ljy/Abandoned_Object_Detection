import torch

from utils.yolov5.utils.general import set_logging,check_img_size,non_max_suppression
from utils.yolov5.utils.torch_utils import select_device
from utils.yolov5.models.experimental import attempt_load


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

    def detect(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        return pred
