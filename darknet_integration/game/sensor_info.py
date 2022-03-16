from typing import Tuple
import carla


class SensorInfo:
    """
    Data object with sensor info needed by other objects
    """

    def __init__(
        self,
        z,
        pitch,
        image_size_x,
        image_size_y,
        fov,
        camera_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.z = z
        self.pitch = pitch
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.fov = fov
        self.camera_distance = camera_distance

    def __str__(self) -> str:
        return f"SensorInfo(\
            \nz={self.z}, \
            \npitch={self.pitch}, \
            \nimage_size_x={self.image_size_x}, \
            \nimage_size_y={self.image_size_y}, \
            \nfov={self.fov}) \
            \ncamera_distance={self.camera_distance})\n"

    @staticmethod
    def get_sensor_info(
        sensor: carla.Sensor,
        camera_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "SensorInfo":

        return SensorInfo(
            round(sensor.parent.bounding_box.extent.y * 2 * 100, 2),
            abs(round(sensor.get_transform().rotation.pitch, 2)),
            int(sensor.attributes["image_size_x"]),
            int(sensor.attributes["image_size_y"]),
            float(sensor.attributes["fov"]),
            camera_distance,
        )
