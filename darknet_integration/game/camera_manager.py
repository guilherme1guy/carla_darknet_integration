from functools import cached_property
from typing import List

import carla
from carla import ColorConverter as cc

from game.camera_parser import CameraParser
from game.sensor_abstraction import SensorAbstraction
from game.stereo_sensor_abstraction import StereoSensorAbstraction
from game.transform_data import TransformData


class CameraManager(object):

    # sensor name should be unique
    _SENSORS = [
        SensorAbstraction("sensor.camera.rgb", cc.Raw, "Camera RGB", {}),
        SensorAbstraction(
            "sensor.camera.rgb",
            cc.Raw,
            "Yolo Sensor",
            {
                "fov": "85",
            },
        ),
        StereoSensorAbstraction(
            "sensor.camera.rgb",
            cc.Raw,
            "Yolo Sensor Stereo",
            (0, 0.635, 0),
            {
                "fov": "85",
                "image_size_x": "800",
                "image_size_y": "600",
            },
        ),
    ]

    def __init__(self, parent_actor, hud, gamma_correction):

        self.sensor: SensorAbstraction = None

        self._parent = parent_actor

        self.hud = hud

        self.transform_index = 0

        self.sensors: List[SensorAbstraction] = []
        self.initialize_sensors(gamma_correction)

        # using getattr to avoid errors when the lidar sensor is not used
        self.camera_parser: CameraParser = CameraParser(
            self.hud.dim, getattr(self, "lidar_range", 50)
        )

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
                0.8 * bound_x,
                0.0 * bound_y,
                1.3 * bound_z,
                attachment=Attachment.Rigid,
                pitch=-30,
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
            self._parent,
            self._camera_transforms[self.transform_index],
            self.camera_parser,
        )

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

    def stop(self):
        self.camera_parser.yolo.stop()

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
            self.surface = None

    def toggle_recording(self):
        self.camera_parser.recording = not self.camera_parser.recording
        self.hud.notification(f"Recording {self.camera_parser.recording}")

    def render(self, display: pygame.Surface):

        surface = self.camera_parser.get_surface()

        if surface is not None:
            display.blit(pygame.transform.scale(surface, display.get_size()), (0, 0))
