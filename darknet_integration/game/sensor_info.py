from typing import Tuple
import carla


class SensorInfo:
    """
    Data object with sensor info needed by other objects
    """

    def __init__(
        self,
        transform,
        rotation,
        image_size_x,
        image_size_y,
        fov,
        camera_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.transform = transform
        self.rotation = rotation
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.fov = fov
        self.camera_distance = camera_distance

    def __str__(self) -> str:
        return f"SensorInfo(\
            \ntransform={self.transform}, \
            \nrotation={self.rotation}, \
            \nimage_size_x={self.image_size_x}, \
            \nimage_size_y={self.image_size_y}, \
            \nfov={self.fov}) \
            \ncamera_distance={self.camera_distance})\n"

    @staticmethod
    def get_sensor_info(
        sensor: carla.Sensor,
        camera_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "SensorInfo":

        transform = (
            0,
            0,
            round(sensor.parent.bounding_box.extent.z * 2, 2),
        )

        parent_t = sensor.parent.get_transform()
        t = sensor.get_transform()
        rotation = (
            round(parent_t.rotation.roll - t.rotation.roll, 2),
            round(parent_t.rotation.pitch - t.rotation.pitch, 2),
            round(parent_t.rotation.yaw - t.rotation.yaw, 2),
        )

        return SensorInfo(
            transform,
            rotation,
            int(sensor.attributes["image_size_x"]),
            int(sensor.attributes["image_size_y"]),
            float(sensor.attributes["fov"]),
            camera_distance,
        )
