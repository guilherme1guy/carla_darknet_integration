import threading
import weakref
from functools import lru_cache
from multiprocessing import Barrier
from threading import Lock
from typing import List, Optional, Tuple

import carla
import numpy as np

from game.camera_parser import CameraParser
from game.sensor_abstraction import SensorAbstraction
from game.sensor_info import SensorInfo
from game.transform_data import TransformData


class StereoSensorAbstraction(SensorAbstraction):
    def __init__(
        self,
        sensor_type: str,
        color_convert: int,
        name: str,
        between_distances: Tuple[float, float, float],
        options: dict,
    ) -> None:
        super().__init__(sensor_type, color_convert, name, options)

        # the original transform will be changed, in example:
        # distances_x = 2
        # so the original transform will be on x = -1 and
        # the new transform will be on x = 1

        # between_distances -> (x_distance, y_distance, z_distance)
        self.between_distances = np.array(between_distances)
        self._other_sensor: Optional[carla.Sensor] = None

        self._image_written_barrier = Barrier(2)
        self._locks = [Lock(), Lock()]

        self.images = [None, None]

    def _modify_transform_data(self, transform_data: TransformData, values: np.ndarray):
        # values is a array with 3 elements: [x,y,z]
        return TransformData(
            transform_data.x + values[0],
            transform_data.y + values[1],
            transform_data.z + values[2],
            transform_data.pitch,
            transform_data.yaw,
            transform_data.roll,
            transform_data.attachment_type,
        )

    def _split_transform_data(self, transform_data):
        split_values = self.between_distances / 2
        negative_split = split_values * -1

        return [
            self._modify_transform_data(transform_data, negative_split),
            self._modify_transform_data(transform_data, split_values),
        ]

    def _create_actor(self, world, transform_data: TransformData, parent):

        transforms = self._split_transform_data(transform_data)

        self._sensor = world.spawn_actor(
            self.blueprint,
            transforms[0].transform,
            attach_to=parent,
            attachment_type=transforms[0].attachment_type,
        )

        self._other_sensor = world.spawn_actor(
            self.blueprint,
            transforms[1].transform,
            attach_to=parent,
            attachment_type=transforms[1].attachment_type,
        )

        self._sensor_info.cache_clear()

        # call print in a thread so CARLA can spawn the sensor in the simulation
        threading.Timer(
            0.1,
            lambda: print(
                f"StereoSensor spawned actors at:\n\t{self._sensor.get_location()}\n\t{self._other_sensor.get_location()}\n\t Diff: {self._sensor.get_location() - self._other_sensor.get_location()}"
            ),
        ).start()

    def _create_listener(self, parser: CameraParser):
        # We need to pass the lambda as a weak reference to avoid a circular reference
        # setup listener for sensor
        weak_ref = weakref.ref(parser)
        weak_self = weakref.ref(self)
        self._sensor.listen(
            lambda image: StereoSensorAbstraction._stereo_listen(
                weak_self, weak_ref, image, 0
            )
        )

        self._other_sensor.listen(
            lambda image: StereoSensorAbstraction._stereo_listen(
                weak_self, weak_ref, image, 1
            )
        )

    def stop(self) -> None:

        if self._sensor is not None:
            self._sensor.stop()

        if self._other_sensor is not None:
            self._other_sensor.stop()

    def destroy(self) -> None:

        self.stop()

        if self._sensor is not None:
            self._sensor.destroy()
            self._sensor = None

        if self._other_sensor is not None:
            self._other_sensor.destroy()
            self._other_sensor = None

    def listen(self, func) -> None:
        # this method is not used in this class, as the sensor.listen() is called when
        # spawning the sensor, only implemented to override super class and avoid errors
        return

    @lru_cache(maxsize=1)
    def _sensor_info(self) -> List[SensorInfo]:
        return [
            SensorInfo.get_sensor_info(self._sensor, self.between_distances),
            SensorInfo.get_sensor_info(self._other_sensor, self.between_distances),
        ]

    @staticmethod
    def _stereo_listen(weak_self, weak_parser, image, id):

        self: StereoSensorAbstraction = weak_self()

        if not self:
            return

        # only one image will be accepted from each sensor
        with self._locks[id]:

            self.images[id] = image

            # wait for all threads to write the image
            self._image_written_barrier.wait()

            # only one thread needs to trigger image parsing
            if id == 0 and None not in self.images:
                CameraParser.parse_stereo_image(
                    weak_parser,
                    self.images,
                    self.sensor_type,
                    self.name,
                    self.color_convert,
                    self._sensor_info(),
                )
