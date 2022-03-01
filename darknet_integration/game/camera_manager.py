from asyncio import FastChildWatcher
from cgitb import reset
from functools import cached_property
import queue
import weakref
from typing import List, Optional

import carla
import numpy as np
import pygame
from carla import ColorConverter as cc
from sensors.yolo_sensor import YoloSensor

from game.sensor_abstraction import SensorAbstraction
from game.transform_data import TransformData


class CameraManager(object):

    # sensor name should be unique
    _SENSORS = [
        SensorAbstraction("sensor.camera.rgb", cc.Raw, "Camera RGB", {}),
        SensorAbstraction(
            "sensor.camera.rgb", cc.Raw, "Yolo Sensor", {}
        ),  # {"fov": "110"},
        # SensorGroup(
        #     [
        #         SensorData("sensor.camera.rgb", cc.Raw, "Yolo Sensor C1", {}),
        #         SensorData("sensor.camera.rgb", cc.Raw, "Yolo Sensor C2", {}),
        #     ]
        # ),
    ]

    def __init__(self, parent_actor, hud, gamma_correction):

        self.sensor: SensorAbstraction = None
        self.surface = None

        self._parent = parent_actor

        self.hud = hud

        self.recording = False

        self.yolo = YoloSensor()

        self.transform_index = 0

        self.sensors: List[SensorAbstraction] = []
        self.initialize_sensors(gamma_correction)

        self.next_sensor()

    def initialize_sensors(self, gamma_correction, skip_blueprint=False):

        if skip_blueprint:
            for sensor in CameraManager._SENSORS:
                self.sensors.append(sensor)
            return

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        for sensor in CameraManager._SENSORS:

            bp = bp_library.find(sensor.sensor_type)

            if sensor.sensor_type.startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(self.hud.dim[0]))
                bp.set_attribute("image_size_y", str(self.hud.dim[1]))

                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))

                for attr_name, attr_value in sensor.options.items():
                    bp.set_attribute(attr_name, attr_value)

            elif sensor.sensor_type.startswith("sensor.lidar"):
                self.lidar_range = 50

                for attr_name, attr_value in sensor.options.items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)

            sensor.blueprint = bp

            self.sensors.append(sensor)

    @cached_property
    def _camera_transforms(self):

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        Attachment = carla.AttachmentType

        return [
            TransformData(
                -2.0 * bound_x,
                0.0 * bound_y,
                2.0 * bound_z,
                pitch=8.0,
                attachment=Attachment.SpringArm,
            ),
            TransformData(
                0.8 * bound_x,
                0.0 * bound_y,
                1.3 * bound_z,
                attachment=Attachment.Rigid,
            ),
            TransformData(
                1.9 * bound_x,
                1.0 * bound_y,
                1.2 * bound_z,
                attachment=Attachment.SpringArm,
            ),
            TransformData(
                -2.8 * bound_x,
                0.0 * bound_y,
                4.6 * bound_z,
                pitch=6.0,
                attachment=Attachment.SpringArm,
            ),
            TransformData(
                -1.0 * bound_x,
                -1.0 * bound_y,
                0.4 * bound_z,
                attachment=Attachment.Rigid,
            ),
        ]

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.sensor, notify=False)

    def set_sensor(
        self, sensor_abstraction: SensorAbstraction, notify=True, index=None
    ):

        if index is not None:
            sensor_abstraction = CameraManager._SENSORS[
                index % len(CameraManager._SENSORS)
            ]

        if self.sensor is not None:
            self.sensor.destroy()
            self.surface = None

        self.sensor = sensor_abstraction.spawn(
            self._parent, self._camera_transforms[self.transform_index]
        )

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        if notify:
            self.hud.notification(sensor_abstraction.name)

    def next_sensor(self):

        index = 0
        if self.sensor is not None:

            # get the index of the current sensor
            index = list(
                filter(
                    lambda x: x[1].name == self.sensor.name,
                    enumerate(CameraManager._SENSORS),
                )
            )[0][0]

            # find the next index (the module operation is to wrap around)
            index = (index + 1) % len(CameraManager._SENSORS)

        self.set_sensor(CameraManager._SENSORS[index])

    def reset(self):

        self.destroy()

        self.transform_index = 1

        self.initialize_sensors(None, skip_blueprint=True)
        self.sensor = CameraManager._SENSORS[0]
        self.set_sensor(self.sensor)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
            self.surface = None

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):

        self: CameraManager = weak_self()

        if not self:
            return

        if self.sensor.sensor_type.startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensor.sensor_type.startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool8),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2
            ] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensor.sensor_type.startswith("sensor.camera.optical_flow"):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:

            image.convert(self.sensor.color_convert)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if "Yolo" in self.sensor.name:

                self.yolo.add_job([array], image.frame)
                self.surface = self.yolo.get_surface()

                if self.surface is None:
                    self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            else:
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)
