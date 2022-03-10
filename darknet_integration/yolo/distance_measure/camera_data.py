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

        self.focus_length = focus_length  # focal length of the camera in cm
        self.skew = skew

        self._fov = 0.0

    @cached_property
    def fx(self):
        return self.focus_length * self.ku

    @cached_property
    def fy(self):
        return self.focus_length * self.kv

    @cached_property
    def kv(self):
        return self.image_width / self.image_height

    @cached_property
    def ku(self):
        return self.image_height / self.image_width

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

    def focus_from_hfov(self, hfov):
        focus_len = abs(int(self.image_width) / (2.0 * math.tan(hfov)))
        focus_len = round(focus_len / 100, 2)

        return focus_len

    def update_properties(self):
        # delete cached properties to update them on
        # the next access
        keys = ["fx", "fy", "kv", "ku", "center_x", "center_y", "v_fov", "h_fov"]
        for key in keys:
            self.__dict__.pop(key, None)
