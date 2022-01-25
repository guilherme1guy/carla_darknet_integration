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
from darknet_integration import yolo

from darknet_integration.yolo import YoloClassifier, YoloDetectionResult


class YoloSensor(object):
    """
    This class receives images from the CARLA simulation and adds them to a queue for later classification
    On another thread the images are transformed into jobs, that run in N worker threads

    A image is discarded when its too old, each thread has its own classifier


    # Better results where found with 1 thread
    """

    THREAD_COUNT = 1

    def __init__(self):

        self.jobs = Queue()
        self.results = []

        self.run = True

        self.threads: List[Thread] = []

        # Initialize worker threads
        for i in range(0, self.THREAD_COUNT):
            worker_thread = Thread(target=self.work, args=(i,))
            worker_thread.start()

            self.threads.append(worker_thread)

    def work(self, thread_id):
        """
        This function is the body of the worker thread, it keeps running until
        self.run is set to false
        """

        print(f"[t{thread_id}] Started thread")

        # initialize classifier for this worker thread
        yolo_classifier = YoloClassifier(
            config="/home/gguy/code/darknet/cfg/yolov3.cfg",
            weights="/home/gguy/code/darknet/yolov3.weights",
            # config="/home/gguy/code/darknet/cfg/yolov3-tiny.cfg",
            # weights="/home/gguy/code/darknet/yolov3-tiny.weights",
            classes="/home/gguy/code/darknet/data/coco.names",
        )

        # worker main loop
        while self.run:

            # try to get a new image
            try:
                # start_time -> when was this image created on the simulation?
                image_array, frame_id, start_time = self.jobs.get(timeout=1)
            except Empty:
                continue

            # current time
            local_start_time = time.time()

            # if the image is older than 0.1, discard it
            if local_start_time - start_time > 0.1:
                continue

            # run job
            self.results.append(self.job(yolo_classifier, image_array))

            # collect data for performance analysis
            end_time = time.time()

            if len(self.results) > 255:
                self.results.pop(0)

            local_delta = round(end_time - local_start_time, 3)
            delta = round(end_time - start_time, 3)
            fps = round(1 / delta, 2)

            print(
                f"[t{thread_id}] Finished job#{frame_id}\
                 localDetal: {local_delta}s delta: {delta}s fps: {fps} qsize: {self.jobs.qsize()}"
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

    def job(self, yolo_classifier, array):
        """
        Actual work that the worker thread must do
        """
        result = yolo_classifier.classify(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

        # ask result object to draw the boxes on the resulting image
        result.draw(result)

        return result

    def add_job(self, array, frame):
        """
        Get image and add it to job queue
        """

        if not self.run:
            return

        self.jobs.put((array, frame, time.time()))
        # print(f"Added job#{frame}, qsize: {self.jobs.qsize()}")

    def get_surface(self):
        """
        Get latest result image and convert to a PyGame Surface for
        displaying
        """

        if len(self.results) < 1:
            return None

        return YoloClassifier.image_to_pygame(self.results[-1].image)
