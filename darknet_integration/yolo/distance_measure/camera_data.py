from functools import cached_property
import math


class CameraData:
    def __init__(
        self,
        height: float,
        angle: float,
        focus_length: float,
        image_width: int,
        image_height: int,
        skew: float = 0,
    ) -> None:

        # image size in pixels
        self.image_width = image_width
        self.image_height = image_height

        self.height: float = height  # height of the camera in (cm)

        # angle (degrees) that the camera is in relation to the ground
        self.angle: float = angle
        self.rad_angle = math.radians(angle)

        self.focus_length = focus_length  # focal length of the camera in mm
        self.skew = skew

    @cached_property
    def center_x(self):
        return self.image_width // 2

    @cached_property
    def center_y(self):
        return self.image_height // 2

    @cached_property
    def v_fov(self):
        return 2 * math.atan(self.image_height / (2 * self.focus_length))

    @cached_property
    def h_fov(self):
        return 2 * math.atan(self.image_width / (2 * self.focus_length))
