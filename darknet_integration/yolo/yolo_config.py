from abc import ABC
from typing import Callable, List

from pytorchyolo import detect, models
from pytorchyolo.utils.utils import load_classes
import torch
import numpy as np
import cv2

from yolo.detection import Detection

from yolov4.tool.darknet2pytorch import Darknet as DarknetV4
from yolov4.tool.torch_utils import do_detect as do_detect_v4


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


class YoloV4Config(YoloConfig):
    def __init__(
        self,
        conf_threshold=0.7,
        nms_threshold=0.6,
        cfg_file="models/yolov4.cfg",
        weights_file="models/yolov4.weights",
        classes_file="models/coco.names",
    ):
        super().__init__(
            conf_threshold,
            nms_threshold,
        )

        self._cfgfile = cfg_file
        self._weightsfile = weights_file
        self._classes = load_classes(classes_file)
        self.use_cuda = torch.cuda.is_available()

        self.img_shape = (0, 0)

        # set seed to get the same colors for each run
        np.random.seed(0)
        self._colors = np.random.uniform(0, 120, size=(len(self._classes), 3))

    def get_model(self):
        model = DarknetV4(self.cfg_file)
        model.load_weights(self.weights_file)

        if self.use_cuda:
            model.cuda()

        return model

    def get_detection_fn(self, model) -> Callable:
        def fn(img):

            self.img_shape = img.shape[:2]

            sized = cv2.resize(img, (model.width, model.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            outputs = do_detect_v4(
                model,
                sized,
                self.conf_threshold,
                self.nms_threshold,
                use_cuda=self.use_cuda,
            )

            return outputs

        return fn

    def get_output_conversion_fn(self) -> Callable:
        def fn(element):
            conv_element = [
                element[0] * self.img_shape[1],
                element[1] * self.img_shape[0],
                element[2] * self.img_shape[1],
                element[3] * self.img_shape[0],
                element[4],
                element[6],
            ]
            return Detection.from_output(conv_element)

        return lambda image_outputs: [
            [fn(element) for element in output[0]] for output in image_outputs
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
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s6", pretrained=True, verbose=False
        )
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
