from functools import lru_cache
import math
from queue import Empty

import numpy as np
from game.sensor_info import SensorInfo
from utils import image_to_pygame
from yolo.distance_measure.camera_data import CameraData
from yolo.distance_measure.ipm_distance_calculator import IPMDistanceCalculator

from sensors.threaded_sensor import ThreadedSensor


class IPMSensor(ThreadedSensor):
    """
    Inverse Perspective Mappin (IPM) calculator sensor, uses the ThreadedSensorJob class to provide asynchronous
    IPM calculation for images, discarding them when its too old.
    """

    THREAD_COUNT = 1

    def __init__(self):

        self.results: list[np.ndarray] = []
        super().__init__()

    @lru_cache
    def get_ipm(self, thread_id):
        # start camera_data with random values, they will be
        # overwritten by the first image
        camera_data = CameraData(200, 30, 2.8, 1280, 720)
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
        ipm = self.get_ipm(thread_id)
        IPMSensor._check_camera_data(ipm, job.extra_data)
        self.results.append(ipm.convert_image(job.cv2_images[0]))

        job.end_job()

        # delete oldest result if we have more than 255
        if len(self.results) > 255:
            self.results.pop(0)

        # print performance info
        print(
            f"[t{str(self)}_{thread_id}] Finished job#{job.frame_id} \
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

        return image_to_pygame(self.results[-1])

    @staticmethod
    def _check_camera_data(ipm: IPMDistanceCalculator, sensor_info: SensorInfo) -> None:

        # camera_data fields to check:
        # height: float,
        # angle: float,
        # focus_length: float,
        # image_width: int,
        # image_height: int,

        camera_data = ipm.camera_data

        conditions = [
            camera_data.height != sensor_info.z,
            camera_data.angle != sensor_info.pitch,
            camera_data.image_width != sensor_info.image_size_x,
            camera_data.image_height != sensor_info.image_size_y,
            camera_data._fov != sensor_info.fov,
            any(np.not_equal(camera_data.camera_distance, sensor_info.camera_distance)),
        ]

        if any(conditions):

            camera_data.height = sensor_info.z
            camera_data.angle = sensor_info.pitch
            camera_data.rad_angle = math.radians(camera_data.angle)
            camera_data.image_width = sensor_info.image_size_x
            camera_data.image_height = sensor_info.image_size_y
            camera_data.focus_length = camera_data.focus_from_hfov(sensor_info.fov)
            camera_data._fov = sensor_info.fov
            camera_data.camera_distance = sensor_info.camera_distance

            camera_data.update_properties()
            ipm.update_properties()

            print(sensor_info)
            print(camera_data)
            print(ipm)

    def __str__(self) -> str:
        return "ipm_t"
