import abc
import time
from queue import Queue
from threading import Thread
from typing import Any, List

import numpy as np

from sensors.threaded_sensor_job import ThreadedSensorJob


class ThreadedSensor(abc.ABC):
    """
    This class receives images from the CARLA simulation and adds them to a queue for later processing.
    On another thread the images are transformed into jobs, that run in N worker threads

    Methods that need to be implemented:
    - work(self, thread_id: int)
    - clear(self)
    - get_surface(self)
    - __str__(self)
    """

    THREAD_COUNT = 1

    def __init__(self):

        self.run = True
        self.threads: List[Thread] = []

        self.jobs: Queue[ThreadedSensorJob] = Queue()

        # Initialize worker threads
        for i in range(0, self.THREAD_COUNT):
            worker_thread = Thread(target=self._work, args=(i,))
            worker_thread.start()

            self.threads.append(worker_thread)

    def _work(self, thread_id: int):
        """
        This function is the body of the worker thread, it keeps running until
        self.run is set to false. It calls the work() method that should be implemented
        by the child classes.
        """
        print(f"[{str(self)}_{thread_id}] Started thread")

        # worker main loop
        while self.run:
            self.work(thread_id)

        print(f"[{str(self)}_{thread_id}] Finished thread")

    @abc.abstractmethod
    def work(self, thread_id: int):
        """
        Function where the worker thread does its work. Only exists in child classes.
        Should return when current job is finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError

    def stop(self):
        """
        Stop sensor execution
        """

        self.run = False

        for thread in self.threads:
            thread.join()

        self.clear()

        print(f"[{str(self)}] Joined all threads")

    def add_job(self, array: List[np.ndarray], frame_id: int, extra_data: Any = None):
        """
        Get image and add it to job queue
        """

        if not self.run:
            return

        self.jobs.put(
            ThreadedSensorJob(array, frame_id, time.time(), extra_data=extra_data)
        )
        # print(f"Added job#{frame}, qsize: {self.jobs.qsize()}")

    @abc.abstractmethod
    def get_surface(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        return "generic_t"
