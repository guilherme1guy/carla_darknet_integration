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
    def __init__(self):

        self.jobs = Queue()
        self.results = []

        self.run = True
        THREAD_COUNT = 1
        self.threads: List[Thread] = []

        for i in range(0, THREAD_COUNT):
            worker_thread = Thread(target=self.work, args=(i,))
            worker_thread.start()

            self.threads.append(worker_thread)

    def work(self, thread_id):

        print(f"[t{thread_id}] Started thread")

        yolo_classifier = YoloClassifier(
            config="/home/gguy/code/darknet/cfg/yolov3.cfg",
            weights="/home/gguy/code/darknet/yolov3.weights",
            # config="/home/gguy/code/darknet/cfg/yolov3-tiny.cfg",
            # weights="/home/gguy/code/darknet/yolov3-tiny.weights",
            classes="/home/gguy/code/darknet/data/coco.names",
        )

        while self.run:
            try:
                array, frame, start_time = self.jobs.get(timeout=1)
            except Empty:
                continue

            local_start_time = time.time()

            if local_start_time - start_time > 0.1:
                continue

            self.results.append(self.job(yolo_classifier, array))
            end_time = time.time()

            if len(self.results) > 255:
                self.results.pop(0)

            local_delta = round(end_time - local_start_time, 3)
            delta = round(end_time - start_time, 3)
            fps = round(1 / delta, 2)

            print(
                f"[t{thread_id}] Finished job#{frame} localDetal: {local_delta}s delta: {delta}s fps: {fps} qsize: {self.jobs.qsize()}"
            )

        print(f"[t{thread_id}] Finished thread")

    def stop(self):

        self.run = False
        self.results.clear()

        for thread in self.threads:
            thread.join()

        print(f"Joined all threads")

    def job(self, yolo, array):

        result = yolo.classify(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        yolo.draw(result)
        return result

    def add_job(self, array, frame):

        if not self.run:
            return

        self.jobs.put((array, frame, time.time()))
        # print(f"Added job#{frame}, qsize: {self.jobs.qsize()}")

    def get_surface(self):

        if len(self.results) < 1:
            return None

        return YoloClassifier.cvimage_to_pygame(self.results[-1].image)
