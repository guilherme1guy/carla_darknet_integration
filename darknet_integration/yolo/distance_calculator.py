from functools import lru_cache


class DistanceCalculator:
    @staticmethod
    @lru_cache
    def get_object_distance(x, y, width, height):
        # x, y = top-left corner of bounding box
        # w, h = width and height of bounding box

        # distance = ((2 * 3.14 * 180) / (width + height * 360)) + C/c

        distance = (2 * 3.14 * 180) / (width + height * 360) * 1000 + 3
        distance = round(distance * 2.54, 1)

        return distance
