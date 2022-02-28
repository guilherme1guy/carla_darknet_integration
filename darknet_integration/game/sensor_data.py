class SensorData:
    def __init__(
        self, sensor_type: str, color_convert: int, name: str, options: dict
    ) -> None:

        self.sensor_type = sensor_type
        self.color_convert = color_convert
        self.name = name
        self.options = options

        self.blueprint = None
