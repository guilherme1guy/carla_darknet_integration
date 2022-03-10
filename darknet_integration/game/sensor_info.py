import carla


class SensorInfo:
    """
    Data object with sensor info needed by other objects
    """

    def __init__(self, z, pitch, image_size_x, image_size_y, fov) -> None:
        self.z = z
        self.pitch = pitch
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.fov = fov

    def __str__(self) -> str:
        return f"SensorInfo(\
            z={self.z}, \
            pitch={self.pitch}, \
            image_size_x={self.image_size_x}, \
            image_size_y={self.image_size_y}, \
            fov={self.fov})"

    @staticmethod
    def get_sensor_info(sensor: carla.Sensor) -> "SensorInfo":

        transform = sensor.get_transform()

        return SensorInfo(
            round(transform.location.z * 100, 2),
            abs(round(transform.rotation.pitch, 2)),
            int(sensor.attributes["image_size_x"]),
            int(sensor.attributes["image_size_y"]),
            float(sensor.attributes["fov"]),
        )
