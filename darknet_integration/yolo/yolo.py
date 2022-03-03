from threading import Lock
from typing import List

import cv2
import numpy as np
from torch import classes

from yolo.detection import Detection
from yolo.yolo_config import YoloConfig

cuda_lock = Lock()


class YoloClassifier(object):
    def __init__(self, yolo_cfg: YoloConfig, conf_threshold=0.5, nms_threshold=0.4):

        self.yolo_cfg = yolo_cfg
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.model = None
        self._load_model()

    def _process_output(self, outs, height, width):

        boxes = []
        confs = []
        classess_ids = []

        for detections in outs:
            for detection in detections:

                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > self.conf_threshold:

                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, bwidth, bheight = box.astype("int")
                    x = int(centerX - (bwidth / 2))
                    y = int(centerY - (bheight / 2))

                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confs.append(float(conf))
                    classess_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.nms_threshold)

        results = []
        for idx in idxs:
            results.append(
                Detection.from_cv2_output(boxes[idx], confs[idx], classess_ids[idx])
            )

        return results

    def detect_objects(self, images: List[np.ndarray]) -> List[List]:

        outputs = []

        for image in images:

            self.model.setInput(
                cv2.dnn.blobFromImage(
                    image,
                    scalefactor=1 / 255.0,
                    size=(416, 416),
                    mean=(0, 0, 0),
                    swapRB=True,
                    crop=False,
                )
            )

            with cuda_lock:

                output = self.model.forward(self.output_layers)

                outputs.append(
                    self._process_output(
                        output,
                        image.shape[0],
                        image.shape[1],
                    )
                )

        return outputs

    def _load_model(self):

        self.model = cv2.dnn.readNet(self.yolo_cfg.weights_file, self.yolo_cfg.cfg_file)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.layers_names = self.model.getLayerNames()
        self.output_layers = [
            self.layers_names[i - 1] for i in self.model.getUnconnectedOutLayers()
        ]

    def classify(self, images: List[np.ndarray]) -> List[np.ndarray]:
        # image must be cv2 image

        # Output is a list with a numpy array for each image
        # with the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        outputs = self.detect_objects(images)

        for index, output in enumerate(outputs):
            for detection in output:
                self.draw_on_image(images[index], detection)

        return images

    def draw_on_image(self, image, detection: Detection):
        """
        Draws the bounding box over the objects that the model detects
        """

        # as a personal choice you can modify this to get distance as accurate as possible:
        # detection.x1 += 150
        # detection.y1 += 100
        # detection.x2 += 200
        # detection.y2 += 200

        label = [
            f"{self.yolo_cfg.classes[detection.class_index]}",
            f"{detection.distance}m - {detection.confidence*100:.2f}%",
        ]
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
            (detection.x1 - 2, detection.y2 + 25 * len(label)),
            (detection.x2 + 2, detection.y2),
            color,
            -1,
        )

        # write label to image
        for idx, line in enumerate(label):
            image = cv2.putText(
                image,
                line,
                (detection.x1 + 2, detection.y2 + 20 * (idx + 1)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                [225, 255, 255],
                1,
            )

        # returns image with bounding box and label drawn on it
        return image
