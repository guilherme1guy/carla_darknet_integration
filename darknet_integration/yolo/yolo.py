import math
from threading import Lock
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pytorchyolo import detect, models

from yolo.detection import Detection
from yolo.distance_measure.ipm_distance_calculator import IPMDistanceCalculator
from yolo.distance_measure.stereo_distance_calculator import StereoDistance
from yolo.yolo_config import YoloConfig

cuda_lock = Lock()


class YoloClassifier(object):
    def __init__(self, yolo_cfg: YoloConfig, conf_threshold=0.7, nms_threshold=0.6):

        self.yolo_cfg = yolo_cfg
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.model = None
        self._load_model()

    def detect_objects(self, images: List[np.ndarray]) -> List[List]:

        outputs = []

        for image in images:
            with cuda_lock:

                output = detect.detect_image(
                    self.model,
                    image,
                    conf_thres=self.conf_threshold,
                    nms_thres=self.nms_threshold,
                )

                outputs.append(output)

        return outputs

    def _load_model(self):

        self.model = models.load_model(
            self.yolo_cfg.cfg_file, self.yolo_cfg.weights_file
        )

    def _detections_from_outputs(self, image_outputs) -> List[List[Detection]]:
        return [
            [Detection.from_output(element) for element in output]
            for output in image_outputs
        ]

    def _map_similar_detections(
        self,
        left_image_detections: List[Detection],
        right_image_detections: List[Detection],
    ):

        for left_det in left_image_detections:

            candidates = []

            for right_det in right_image_detections:

                # object should not be used by other
                if right_det.similar_detection != None:
                    continue

                # objects should be from the same class
                if right_det.class_index != left_det.class_index:
                    continue

                # object from the right picture should be closer to
                # the left border of the image
                if right_det.x1 > left_det.x1:
                    continue

                candidates.append(right_det)

            if len(candidates) > 0:
                left_det.similar_detection = left_det.most_similar(candidates)
                # set inverse relation
                left_det.similar_detection.similar_detection = left_det

    def _stereo_classify(
        self,
        images: List[np.ndarray],
        detections: List[List[Detection]],
        ipm: Optional[IPMDistanceCalculator] = None,
    ) -> Tuple[List[np.ndarray], List[List[Detection]]]:

        # as calculating the ipm for each point is expensive, but
        # in a batch is not so much, we will first calculate all ipm
        # results

        if ipm:
            ipm_results = [
                list(
                    map(
                        lambda x: ipm.inverse_projection(*x.distance_pivot),
                        detections[0],
                    )
                ),
                list(
                    map(
                        lambda x: ipm.inverse_projection(*x.distance_pivot),
                        detections[1],
                    )
                ),
            ]

            # insert ipm position into detections
            for image_index, image_detections in enumerate(detections):
                for index, detection in enumerate(image_detections):
                    point = ipm_results[image_index][index].flatten()
                    detection.ipm_x = round(point[0], 2)
                    detection.ipm_y = round(point[1], 2)

        # map similar detections
        self._map_similar_detections(detections[0], detections[1])

        for image_index, image_detections in enumerate(detections):

            for detection in image_detections:

                if ipm and detection.similar_detection:

                    detection.stereo_distance = StereoDistance.distance(
                        camera_distance=ipm.camera_data.camera_distance[1],
                        image_width=images[0].shape[1],
                        fov=ipm.camera_data._fov,
                        x1=detection.distance_pivot[0],
                        x2=detection.similar_detection.distance_pivot[0],
                    )

                    detection.ipm_stereo_distance = StereoDistance.distance(
                        camera_distance=ipm.camera_data.camera_distance[1],
                        image_width=images[0].shape[1],
                        fov=ipm.camera_data._fov,
                        x1=detection.ipm_x,
                        x2=detection.similar_detection.ipm_x,
                    )

                    detection.ipm_distance = IPMDistanceCalculator.distance_from_points(
                        detection.ipm_x, detection.ipm_y
                    )

                self.draw_on_image(images[image_index], detection)

        return images, detections

    def _classify(
        self,
        image: np.ndarray,
        detections: List[Detection],
        ipm: Optional[IPMDistanceCalculator] = None,
    ) -> Tuple[List[np.ndarray], List[List[Detection]]]:

        ipm_results = []
        if ipm:
            ipm_results = [
                ipm.inverse_projection(*x.distance_pivot).flatten() for x in detections
            ]

        for index, detection in enumerate(detections):

            if ipm:
                x, y, _ = ipm_results[index]
                detection.ipm_distance = IPMDistanceCalculator.distance_from_points(
                    x, y
                )
                detection.ipm_x = round(x, 2)
                detection.ipm_y = round(y, 2)

            self.draw_on_image(image, detection)

        return [image], [detections]

    def classify(
        self, images: List[np.ndarray], ipm: Optional[IPMDistanceCalculator] = None
    ) -> Tuple[List[np.ndarray], List[List[Detection]]]:
        # image must be cv2 image

        # Output is a list with a numpy array for each image
        # with the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        outputs = self.detect_objects(images)
        detections = self._detections_from_outputs(outputs)

        if len(outputs) > 1:
            return self._stereo_classify(images, detections, ipm)
        else:
            return self._classify(images[0], detections[0], ipm)

    def draw_on_image(self, image, detection: Detection):
        """
        Draws the bounding box over the objects that the model detects
        """

        label = [
            f"{detection.x1}, {detection.y1}",
            f"{round(detection.confidence*100, 2)}%: {self.yolo_cfg.classes[detection.class_index]}",
        ]

        if detection.y2 >= image.shape[0] // 2:
            label += [
                f"({detection.ipm_x}x, {detection.ipm_y}y)",
                f"simple: {detection.simple_distance} m",
                f"ipm: {detection.ipm_distance} m",
            ]

            if detection.similar_detection is not None:
                label += [
                    f"stereo: {detection.stereo_distance} m",
                    f"s_ipm: {detection.ipm_stereo_distance} m",
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

        draw_above = detection.y1 - 25 * len(label) > 0

        if draw_above:
            # draw rectangle for label
            cv2.rectangle(
                image,
                (detection.x1 - 2, detection.y1 - 25 * len(label)),
                (detection.x2 + 2, detection.y1),
                color,
                -1,
            )

            # write label to image
            max_idx = len(label)
            for idx, line in enumerate(label):
                image = cv2.putText(
                    image,
                    line,
                    (detection.x1 + 2, detection.y1 - 20 * (max_idx - idx)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    [225, 255, 255],
                    1,
                )

        else:
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
                    (detection.x1 + 2, detection.y2 + 20 * (idx - 1)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    [225, 255, 255],
                    1,
                )

        # returns image with bounding box and label drawn on it
        return image
