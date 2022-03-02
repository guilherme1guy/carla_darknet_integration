from typing import Any, List, Optional, Tuple
import weakref
import numpy as np
import pygame

from sensors.yolo_sensor import YoloSensor


class CameraParser:
    def __init__(self, dim: Tuple[int, int], lidar_range: float) -> None:

        self.dim: Tuple[int, int] = dim
        self.lidar_range = lidar_range
        self.yolo = YoloSensor()

        self.recording = False

        self.surface: Optional[pygame.Surface] = None

    def get_weak_self(self):
        weak_self = weakref.ref(self)
        return weak_self

    def get_surface(self):
        return self.surface

    def set_surface(self, surface: pygame.Surface):
        self.surface = surface

    @staticmethod
    def _parse_lidar(
        image,
        dim: Tuple[int, int],
        lidar_range: float,
    ) -> pygame.Surface:

        points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        lidar_data = np.array(points[:, :2])

        lidar_data *= min(dim) / (2.0 * lidar_range)
        lidar_data += (0.5 * dim[0], 0.5 * dim[1])

        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        lidar_img_size = (dim[0], dim[1], 3)

        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        return pygame.surfarray.make_surface(lidar_img)

    @staticmethod
    def _parse_dvs(image) -> pygame.Surface:
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
        dvs_img[dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2] = 255

        return pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))

    @staticmethod
    def _parse_optical_flow(image) -> pygame.surface:
        image = image.get_color_coded_flow()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def _rgb_image_to_array(image, color_convert):
        image.convert(color_convert)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return array

    @staticmethod
    def _parse_rgb_camera(
        images: List[Any], color_convert, yolo: Optional[YoloSensor] = None
    ) -> pygame.surface:

        arrays = [
            CameraParser._rgb_image_to_array(image, color_convert) for image in images
        ]

        if yolo is not None:

            yolo.add_job(arrays, images[0].frame)
            surface = yolo.get_surface()

            if surface is not None:
                return surface

        return pygame.surfarray.make_surface(arrays[0].swapaxes(0, 1))

    @staticmethod
    def parse_image(weak_self, image, sensor_type, sensor_name, color_convert):

        self: CameraParser = weak_self()

        if not self:
            return

        if sensor_type.startswith("sensor.lidar"):
            new_surface = CameraParser._parse_lidar(image, self.dim, self.lidar_range)

        elif sensor_type.startswith("sensor.camera.dvs"):
            new_surface = CameraParser._parse_dvs(image)

        elif sensor_type.startswith("sensor.camera.optical_flow"):
            new_surface = CameraParser._parse_optical_flow(image)

        else:
            new_surface = CameraParser._parse_rgb_camera(
                [image], color_convert, (self.yolo if "Yolo" in sensor_name else None)
            )

        self.set_surface(new_surface)

        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)

    @staticmethod
    def parse_stereo_image(weak_self, images, sensor_type, sensor_name, color_convert):

        self: CameraParser = weak_self()

        if not self:
            return

        # this methoud is only used in yolo mode
        new_surface = CameraParser._parse_rgb_camera(images, color_convert, self.yolo)

        self.set_surface(new_surface)

        if self.recording:
            for index, image in enumerate(images):
                image.save_to_disk(f"_out/{'%08d' % image.frame}_{index}")
