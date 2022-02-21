#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


from threading import Lock
from typing import List

import cv2
import numpy as np
import pygame
from pytorchyolo import detect, models

from yolo.detection import Detection
from yolo.yolo_config import YoloConfig


class YoloClassifier(object):
    def __init__(self, yolo_cfg: YoloConfig, conf_threshold=0.5, nms_threshold=0.4):

        self.yolo_cfg = yolo_cfg
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self._lock = Lock()

        self.model = models.load_model(
            self.yolo_cfg.cfg_file, self.yolo_cfg.weights_file
        )

        self.model.eval()

    def detect_objects(self, image) -> List[List]:

        output = None

        with self._lock:
            output = detect.detect_image(
                self.model,
                image,
                conf_thres=self.conf_threshold,
                nms_thres=self.nms_threshold,
            )

        return output

    def classify(self, image):
        # image must be cv2 image

        # Output is a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        output = self.detect_objects(image)

        for out in output:

            detection = Detection.from_output(out)

            self.draw_on_image(image, detection)
            # distance = self.distance_calculator.calc(*detection.bounding_box)

        return image

    def draw_on_image(self, image, detection: Detection):
        """
        Draws the bounding box over the objects that the model detects
        """

        # as a personal choice you can modify this to get distance as accurate as possible:
        # detection.x1 += 150
        # detection.y1 += 100
        # detection.x2 += 200
        # detection.y2 += 200

        label = self.yolo_cfg.classes[detection.class_index]
        color = self.yolo_cfg.colors[detection.class_index]

        # draw rectangle around detected object
        image = cv2.rectangle(
            image,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            1,
        )

        # draw rectangle for label
        cv2.rectangle(
            image,
            (detection.x1 - 2, detection.y2 + 25),
            (detection.x2 + 2, detection.y2),
            color,
            -1,
        )

        # write label to image
        image = cv2.putText(
            image,
            label,
            (detection.x1 + 2, detection.y2 + 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [225, 255, 255],
            1,
        )

        # returns image with bounding box and label drawn on it
        return image

    @staticmethod
    def load_image_file(filename):
        image = cv2.imread(filename)

        return image

    @staticmethod
    def load_image_pygame(surface):
        # based on https://gist.github.com/jpanganiban/3844261
        # and https://stackoverflow.com/questions/53101698/how-to-convert-a-pygame-image-to-open-cv-image
        # and https://www.reddit.com/r/pygame/comments/gldeqs/pygamesurfarrayarray3d_to_image_cv2/

        view = pygame.surfarray.array3d(surface)
        view = view.transpose([1, 0, 2])
        img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        return img_bgr

    @staticmethod
    def image_to_pygame(image) -> pygame.Surface:

        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = np.rot90(np.fliplr(im))
        surface = pygame.surfarray.make_surface(im)

        return surface
