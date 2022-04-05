import math


class StereoDistance:
    @staticmethod
    def distance(
        camera_distance: float, image_width: float, fov: float, x1: float, x2: float
    ) -> float:

        # fov is in degrees, convert to radians to get correct result
        tan = math.tan(math.radians(fov) / 2)
        delta_x = abs(x1 - x2)

        distance = (camera_distance * image_width) / (2 * tan * delta_x)

        return round(distance, 3)
