from abc import ABC
from typing import Callable, List

from pytorchyolo import detect, models
from pytorchyolo.utils.utils import load_classes
import torch
import numpy as np

from yolo.detection import Detection


class YoloConfig(ABC):
    def __init__(self, conf_threshold=0.7, nms_threshold=0.6):
        self._cfgfile = ""
        self._weightsfile = ""
        self._classes = [""]
        self._colors = []

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    @property
    def cfg_file(self):
        return self._cfgfile

    @property
    def weights_file(self):
        return self._weightsfile

    @property
    def classes(self):
        return self._classes

    @property
    def colors(self):
        return self._colors

    def get_model(self):
        raise NotImplementedError

    def get_detection_fn(self, model) -> Callable:
        raise NotImplementedError

    def get_output_conversion_fn(self) -> Callable:
        raise NotImplementedError


class YoloV3Config(YoloConfig):
    def __init__(
        self,
        conf_threshold=0.7,
        nms_threshold=0.6,
        cfg_file="models/yolov3.cfg",
        weights_file="models/yolov3.weights",
        classes_file="models/coco.names",
    ):
        super().__init__(conf_threshold, nms_threshold)

        self._cfgfile = cfg_file
        self._weightsfile = weights_file
        self._classes = load_classes(classes_file)

        # set seed to get the same colors for each run
        np.random.seed(0)
        self._colors = np.random.uniform(0, 120, size=(len(self._classes), 3))

    def get_model(self):
        return models.load_model(self.cfg_file, self.weights_file)

    def get_detection_fn(self, model_obj) -> Callable:
        def fn(img):

            return detect.detect_image(
                model_obj,
                img,
                conf_thres=self.conf_threshold,
                nms_thres=self.nms_threshold,
            )

        return fn

    def get_output_conversion_fn(self) -> Callable:
        return lambda image_outputs: [
            [Detection.from_output(element) for element in output]
            for output in image_outputs
        ]


class YoloV5Config(YoloConfig):
    def __init__(
        self, conf_threshold=0.7, nms_threshold=0.6, classes_file="models/coco.names"
    ):
        super().__init__(
            conf_threshold,
            nms_threshold,
        )

        self._classes = load_classes(classes_file)

        # set seed to get the same colors for each run
        np.random.seed(0)
        self._colors = np.random.uniform(0, 120, size=(len(self._classes), 3))

    def get_model(self):
        model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True)
        model.conf = self.conf_threshold
        model.ioy = self.nms_threshold

        return model

    def get_detection_fn(self, model) -> Callable:
        return model

    def get_output_conversion_fn(self) -> Callable:
        return lambda image_outputs: [
            [Detection.from_output(element) for element in output.xyxy[0]]
            for output in image_outputs
        ]
