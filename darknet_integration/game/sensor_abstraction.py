import typing
from typing import Optional
import weakref

import carla
from game.camera_parser import CameraParser

from game.transform_data import TransformData


class SensorAbstraction:
    def __init__(
        self, sensor_type: str, color_convert: int, name: str, options: dict
    ) -> None:

        self.sensor_type = sensor_type
        self.color_convert = color_convert
        self.name = name
        self.options = options

        self.blueprint = None

        self._sensor: Optional[carla.Sensor] = None

    def spawn(
        self, parent: carla.Actor, transform_data: TransformData, parser: CameraParser
    ) -> "SensorAbstraction":

        world: carla.World = typing.cast(carla.World, parent.get_world())

        self._sensor = world.spawn_actor(
            self.blueprint,
            transform_data.transform,
            attach_to=parent,
            attachment_type=transform_data.attachment_type,
        )

        # We need to pass the lambda as a weak reference to avoid a circular reference
        # setup listener for sensor
        weak_ref = weakref.ref(parser)
        self.listen(
            lambda image: CameraParser.parse_image(
                weak_ref, image, self.sensor_type, self.name, self.color_convert
            )
        )

        return self

    def stop(self) -> None:

        if self._sensor is not None:
            self._sensor.stop()

    def destroy(self) -> None:

        if self._sensor is not None:
            self.stop()
            self._sensor.destroy()
            self._sensor = None

    def listen(self, func) -> None:

        if self._sensor is not None:
            self._sensor.listen(func)
