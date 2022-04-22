from functools import cached_property, lru_cache
import math
from typing import List, Tuple
import numpy as np

from .camera_data import CameraData


class IPMDistanceCalculator:

    # u is always column number (ie: 0 to 1280 (in 720p))
    # v is always row number (ie: 0 to 720 (in 720p))

    def __init__(self, camera_data: CameraData) -> None:

        self.camera_data = camera_data

        self.update_properties()

    @cached_property
    def rotation_matrix(self):

        return self.rotation_from_euler(*self.camera_data.rotation)

    @staticmethod
    def rotation_from_euler(roll, pitch, yaw):

        roll_rad, pitch_rad, yaw_rad = (
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
        )

        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)

        RX = np.array(
            [
                [1, 0, 0],
                [0, cos_pitch, -sin_pitch],
                [0, sin_pitch, cos_pitch],
            ]
        )

        cos_roll = math.cos(roll_rad)
        sin_roll = math.sin(roll_rad)

        RZ = np.array(
            [
                [cos_roll, 0, sin_roll],
                [0, 1, 0],
                [-sin_roll, 0, cos_roll],
            ]
        )

        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        RY = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ]
        )

        return RX @ RY @ RZ

    @cached_property
    def translation_matrix(self):

        T = np.zeros((3, 1))
        T[:3, 0] = self.camera_data.translation

        return T

    @cached_property
    def camera_parameter_matrix(self):

        fx = self.camera_data.fx
        fy = self.camera_data.fy
        s = self.camera_data.skew
        u0 = self.camera_data.center_x
        v0 = self.camera_data.center_y

        return np.array(
            [
                [fx, s, u0],
                [0, fy, v0],
                [0, 0, 1],
            ]
        )

    @cached_property
    def _complete_P_matrix(self):

        extrinsic = np.zeros((3, 4))
        extrinsic[:3, :3] = self.rotation_matrix
        extrinsic[:3, 3] = self.translation_matrix[:3, 0]

        intrinsic = self.camera_parameter_matrix

        return intrinsic @ extrinsic

    @cached_property
    def P_matrix(self):
        # simplify P removing Y axis, since Y = 0
        return np.delete(self._complete_P_matrix, 1, 1)

    @lru_cache
    def convert_point(self, u: int, v: int):
        # [u] = [P00, P01, P02][X]
        # [v] = [P10, P11, P12][Z]
        # [1] = [P20, P21, P22][1]

        # 1. P00*X + P01*Z + P02 = u
        # 2. P10*X + P11*Z + P12 = v
        # 3. P20*X + P21*Z + P22 = 1

        result = np.linalg.solve(self.P_matrix, [u, v, 1])

        # result is [x, z]
        result = (result / result[2])[:-1]

        return result

    def convert_points(self, points: List[Tuple[float, float]]):

        # points is a list of [u, v]

        # simillart to convert_matrix, but only with defined points
        # it is much faster to do a batch call to np.linalg.solve

        if len(points) == 0:
            return []

        result = np.linalg.solve(
            self._P_matrix_array_cache(1, len(points)),
            [[*point, 1] for point in points],
        )
        result = np.array([(r / r[2])[:-1] for r in result])

        return result

    @lru_cache(maxsize=15)
    def _P_matrix_array_cache(self, u_size: int, v_size: int):
        return [self.P_matrix for i in range(u_size * v_size)]

    @lru_cache(maxsize=2)
    def _b_matrix_array_cache(self, u_size: int, v_size: int):
        return [[u, v, 1] for v in range(v_size) for u in range(u_size)]

    def convert_matrix(self, u_size: int, v_size: int) -> np.ndarray:

        # u_size: number of columns (image width)
        # v_size: number of rows (image height)

        result = np.linalg.solve(
            self._P_matrix_array_cache(u_size, v_size),
            self._b_matrix_array_cache(u_size, v_size),
        )
        # result is [x, z], but we have [x, z, w]
        # to convert we need to use [x/w, z/w]
        result = np.array([(r / r[2])[:-1] for r in result])

        # each result is linked to a pixel in the image:
        # [(u, v)] -> is walking the col number (u) and then the row number (v)
        # [(0, 0), (1, 0), ... (u_size - 1, 0), (0, 1)... (u_size - 1, v_size - 1)]

        # b == self._b_matrix_array_cache(u_size, v_size)
        # b[u_size - 1] == [u_size - 1,   0,   1]
        # b[u_size] == [0, 1, 1])

        # which means that the result matrix is a flattened array
        # of [[1280], [1280], ... (720 times) ... [1280]]

        # reshape accordingly
        result = np.reshape(result, (v_size, u_size, 2))

        # result should be accessed as result[row][col] (row -> v; col -> u)
        # ie: result[0][0], result[719][1279], result[v][u]
        return result

    def convert_image(self, image: np.ndarray):

        shape = [image.shape[0], image.shape[1], image.shape[2]]
        new_image = np.zeros(shape, dtype=np.uint8)
        center_x = new_image.shape[0] // 2
        center_z = new_image.shape[1] // 2

        # image.shape[0] is the height of the image (number of rows)
        # image.shape[1] is the width of the image (number of columns)

        results = self.convert_matrix(image.shape[1], image.shape[0])
        u = 0  # column number (ie: 0 to 1280 (in 720p))
        v = 0  # row number (ie: 0 to 720 (in 720p))

        new_height = new_image.shape[0]
        new_width = new_image.shape[1]

        max_x = np.max([max(abs(min(a[0])), max(a[0])) for a in results])
        max_z = np.max([max(abs(min(a[1])), max(a[1])) for a in results])

        for v in range(image.shape[0]):  # for v in heigth (0 to 720p)
            for u in range(image.shape[1]):  # for u in col (0 to 1280)

                # x is the horizontal distance to the point
                # z is the depth of the point
                x, z = results[v, u]

                # then we need to convert the x to the new row number
                # and z to the new column number

                try:
                    # new_v = center_x + -1 * int(self.camera_data.fx * x / z)
                    # new_u = center_z + int(new_width * self.camera_data.fy * 1 / z)

                    new_v = center_x + int(new_height * (x / max_x))
                    new_u = center_z + int(new_width * (z / max_z))

                    if (
                        0 <= new_v < new_image.shape[0]
                        and 0 <= new_u < new_image.shape[1]
                    ):
                        new_image[new_v, new_u] = image[v, u]
                except:
                    # if we get an error we should continue to the next pixel
                    continue

        return self.filter(new_image)

    def filter(self, image: np.ndarray):
        v_mid = image.shape[0] // 2
        for u in range(1, image.shape[1] - 1):
            if (
                image[v_mid, u, 0] == 0
                and image[v_mid, u, 1] == 0
                and image[v_mid, u, 2] == 0
            ):
                image[:, u] = image[:, u - 1]

        return image

    def update_properties(self):

        # delete cached attributes
        keys = [
            "rotation_matrix",
            "translation_matrix",
            "camera_parameter_matrix",
            "P_matrix",
            "_complete_P_matrix",
        ]
        for key in keys:
            self.__dict__.pop(key, None)

        # delete lru_cache
        self._b_matrix_array_cache.cache_clear()
        self._P_matrix_array_cache.cache_clear()
        self.convert_point.cache_clear()

    def __str__(self) -> str:
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        return (
            f"\n--------------\nRotation matrix: \n{self.rotation_matrix}"
            + f"\n--------------\nTranslation matrix: \n{self.translation_matrix}"
            + f"\n--------------\nCamera parameter matrix: \n{self.camera_parameter_matrix}"
            + f"\n--------------\nT @ R (extrinsic): \n{None}"
            + f"\n--------------\nComplete P matrix: \n{self._complete_P_matrix}"
            + f"\n--------------\nP matrix: \n{self.P_matrix}"
        )
