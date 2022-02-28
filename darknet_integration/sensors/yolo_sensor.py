import collections
import io
import math
from typing import List
import weakref
import queue

import carla
import cv2
import json
import time
import numpy as np

from threading import Thread, local
from queue import Empty, Queue
from yolo.yolo_config import YoloV3Config

from yolo.yolo import YoloClassifier
from yolo.yolo_job import YoloJob
from utils import image_to_pygame

from sensors.camera_sensor import CameraSensor


class YoloSensor(object):
    """
    This class receives images from the CARLA simulation and adds them to a queue for later classification
    On another thread the images are transformed into jobs, that run in N worker threads

    A image is discarded when its too old, each thread has its own classifier


    # Better results where found with 1 thread
    """

    THREAD_COUNT = 1

    def __init__(self):

        self.jobs: Queue[YoloJob] = Queue()
        self.results: list[List[np.ndarray]] = []

        self.run = True

        self.threads: List[Thread] = []

        # Initialize worker threads
        for i in range(0, self.THREAD_COUNT):
            worker_thread = Thread(target=self.work, args=(i,))
            worker_thread.start()

            self.threads.append(worker_thread)

    def work(self, thread_id: int):
        """
        This function is the body of the worker thread, it keeps running until
        self.run is set to false
        """

        print(f"[t{thread_id}] Started thread")

        # initialize classifier for this worker thread
        yolo_classifier = YoloClassifier(YoloV3Config())

        # worker main loop
        while self.run:

            # try to get a new image
            try:
                # start_time -> when was this image created on the simulation?
                job = self.jobs.get(timeout=1)
            except Empty:
                continue

            # if the image is older than 0.1, discard it
            if not job.start_and_check():
                continue

            # run job
            self.results.append(self.run_job(yolo_classifier, job))

            job.end_job()

            # delete oldest result if we have more than 255
            if len(self.results) > 255:
                self.results.pop(0)

            print(
                f"[t{thread_id}] Finished job#{job.frame_id} \
                localDetal: {job.local_delta}s delta: {job.delta}s fps: {job.fps} qsize: {self.jobs.qsize()}"
            )

        print(f"[t{thread_id}] Finished thread")

    def stop(self):
        """
        Stop sensor execution
        """

        self.run = False
        self.results.clear()

        for thread in self.threads:
            thread.join()

        print(f"Joined all threads")

    def run_job(
        self, yolo_classifier: YoloClassifier, job: YoloJob
    ) -> List[np.ndarray]:
        """
        Actual work that the worker thread must do
        """
        return yolo_classifier.classify(job.cv2_images)

    def add_job(self, array: List[np.ndarray], frame_id: int):
        """
        Get image and add it to job queue
        """

        if not self.run:
            return

        self.jobs.put(YoloJob(array, frame_id, time.time()))
        # print(f"Added job#{frame}, qsize: {self.jobs.qsize()}")

    def get_surface(self):
        """
        Get latest result image and convert to a PyGame Surface for
        displaying
        """

        if len(self.results) < 1:
            return None

        latest_result = self.results[-1]

        return image_to_pygame(latest_result[-1])
