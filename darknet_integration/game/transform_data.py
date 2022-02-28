from functools import cached_property

import carla


class TransformData:
    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
        attachment: int = 0,
    ) -> None:

        self.x = x
        self.y = y
        self.z = z

        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        self.attachment_type = attachment

    @cached_property
    def transform(self):
        return carla.Transform(
            carla.Location(self.x, self.y, self.z),
            carla.Rotation(self.pitch, self.yaw, self.roll),
        )
