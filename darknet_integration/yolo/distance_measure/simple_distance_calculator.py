from functools import lru_cache


class DistanceCalculator:
    @staticmethod
    @lru_cache
    def get_object_distance(x, y, width, height):
        # returns distance in meters
        # x, y = top-left corner of bounding box  <--- not used
        # w, h = width and height of bounding box

        # from master dissertation:
        # distance = ((2 * 3.14 * 180) / (width + height * 360)) + C/c
        # where c is the real object height and C is the object height in the image
        # adapted (its not even close to correct):
        # distance = ((2 * 3.14 * 180) / (width + height * 360)) + (height / 6.7)

        # https://github.com/paul-pias/Object-Detection-and-Distance-Measurement/issues/3#issuecomment-581238883
        # the '* 1000' is for unit selection
        # the +3 is for defining a minimum distance bias (possibly to avoid a dead zone in the image?)
        distance = (360 * 3.14) / (width + height * 360) * 1000 + 3
        # converts from inches (?) to m
        distance *= 2.54 / 10

        return round(distance, 1)
