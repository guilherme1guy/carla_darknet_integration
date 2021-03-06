from functools import lru_cache
from queue import Empty
from typing import List, Tuple

import cv2
import numpy as np
from sensors.ipm_sensor import IPMSensor
from local_utils import image_to_pygame
from yolo.detection import Detection
from yolo.distance_measure.camera_data import CameraData
from yolo.distance_measure.ipm_distance_calculator import IPMDistanceCalculator
from yolo.yolo import YoloClassifier
from yolo.yolo_config import YoloConfig, YoloV3Config, YoloV5Config, YoloV4Config

from sensors.threaded_sensor import ThreadedSensor


class YoloSensor(ThreadedSensor):
    """
    Yolo detection sensor, uses the ThreadedSensorJob class to provide asynchronous
    object detection, discarding a image when its too old.

    Each thread has its own classifier

    # Better results where found with 1 thread
    """

    THREAD_COUNT = 1

    def __init__(self, yolo_version="v3"):

        # default is YoloV3
        if yolo_version == "v3":
            self.yolo_cfg: YoloConfig = YoloV3Config()
        elif yolo_version == "v4":
            self.yolo_cfg: YoloConfig = YoloV4Config()
        elif yolo_version == "v5":
            self.yolo_cfg: YoloConfig = YoloV5Config()
        else:
            raise ValueError(f"Unknown yolo version: {yolo_version}")

        self.results: list[Tuple[List[np.ndarray], List[List[Detection]]]] = []
        super().__init__()

    @lru_cache
    def yolo_classifier(self, thread_id):
        # This function is cached, so the classifier will not be created
        # everytime. It is important to have thread_id, even if not used
        # in the function, as it will be used by lru_cache to cache a classifier
        # for each thread based on it
        return YoloClassifier(self.yolo_cfg)

    @lru_cache
    def get_ipm(self, thread_id):
        # start camera_data with random values, they will be
        # overwritten by the first image
        camera_data = CameraData((0, 0, 200), (0, 30, 0), 2.8, 1280, 720)
        return IPMDistanceCalculator(camera_data)

    def work(self, thread_id: int):

        # try to get a new image
        try:
            job = self.jobs.get(timeout=1)
        except Empty:
            return

        # if the image is older than 0.1, discard it
        if not job.start_and_check():
            return

        # run job
        # obtains classifier
        yolo_classifier = self.yolo_classifier(thread_id)
        ipm = self.get_ipm(thread_id)

        IPMSensor._check_camera_data(ipm, job.extra_data)

        self.results.append(yolo_classifier.classify(job.cv2_images, ipm))

        job.end_job()

        # delete oldest result if we have more than 255
        if len(self.results) > 255:
            self.results.pop(0)

        # print performance info
        print(
            f"[{str(self)}_{thread_id}] Finished job#{job.frame_id} \
            localDetal: {job.local_delta}s \
            ({int(job.local_delta/job.delta * 100)}%) \
            delta: {job.delta}s \
            fps: {job.fps} qsize: {self.jobs.qsize()}"
        )

    def clear(self):
        self.results.clear()

    def get_surface(self):
        """
        Get latest result image and convert to a PyGame Surface for
        displaying
        """

        if len(self.results) < 1:
            return None

        latest_result, latest_detections = self.results[-1]

        if len(latest_result) > 1:
            unified_image = np.concatenate(latest_result, axis=1)
            resized = cv2.resize(unified_image, dsize=(1900, 1020))
            return image_to_pygame(resized)

        return image_to_pygame(np.concatenate(latest_result, axis=1))

    def __str__(self) -> str:
        return "yolo_t"
