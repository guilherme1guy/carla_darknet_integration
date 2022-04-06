from functools import cached_property
import math
from typing import Tuple


class CameraData:
    def __init__(
        self,
        translation: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        focus_length: float,
        image_width: int,
        image_height: int,
        skew: float = 0,
        camera_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:

        # image size in pixels
        self.image_width = image_width
        self.image_height = image_height

        # translation from camera to world (in meters)
        # (x, y, z)
        self.translation = translation

        # rotation in degrees that the camera is in relation to the ground
        # (roll, pitch, yaw)
        self.rotation = rotation
        self.rad_rotation = (
            math.radians(self.rotation[0]),
            math.radians(self.rotation[1]),
            math.radians(self.rotation[2]),
        )

        self.focus_length = focus_length
        self._fov = self.hfov_from_focus(self.focus_length, self.image_width)
        self.skew = skew

        self.camera_distance = camera_distance

    @cached_property
    def fx(self):
        return self.focus_length  # * self.ku

    @cached_property
    def fy(self):
        return self.focus_length  # * self.kv

    @cached_property
    def kv(self):
        return self.image_height

    @cached_property
    def ku(self):
        return self.image_width

    @cached_property
    def center_x(self):
        return self.image_width // 2

    @cached_property
    def center_y(self):
        return self.image_height // 2

    @cached_property
    def h_fov(self):
        return self._fov

    @staticmethod
    def focus_from_hfov(hfov, width):
        focus_len = width / (2.0 * math.tan(math.radians(hfov / 2))) / 1

        return focus_len

    @staticmethod
    def hfov_from_focus(focus, width):
        hfov = math.degrees(2 * math.atan(width / (2 * focus * 1)))

        return hfov

    @staticmethod
    def vfov_from_hfov(hfov, height, width):
        hfov_rad = math.radians(hfov)
        aspect_ratio = height / width
        vfov = 2 * math.atan(math.tan(hfov_rad / 2) * aspect_ratio)

        return math.degrees(vfov)

    @staticmethod
    def deg_rotation_to_rad(rotation):
        return (
            math.radians(rotation[0]),
            math.radians(rotation[1]),
            math.radians(rotation[2]),
        )

    def update_properties(self):
        # delete cached properties to update them on
        # the next access
        keys = ["fx", "fy", "kv", "ku", "center_x", "center_y", "v_fov", "h_fov"]
        for key in keys:
            self.__dict__.pop(key, None)

    def __str__(self) -> str:
        return f"CameraData(\
            \ntranslation={self.translation},\
            \nrotation={self.rotation},\
            \nrad_rotation={self.rad_rotation},\
            \nfocus_length={self.focus_length},\
            \nimage_width={self.image_width},\
            \nimage_height={self.image_height},\
            \nskew={self.skew},\
            \nfx={self.fx},\
            \nfy={self.fy},\
            \nkv={self.kv},\
            \nku={self.ku},\
            \ncenter_x={self.center_x},\
            \ncenter_y={self.center_y},\
            \nh_fov={self.h_fov},\
            )\n"
