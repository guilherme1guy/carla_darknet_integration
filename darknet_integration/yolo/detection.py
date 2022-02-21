from functools import cached_property
from typing import List

from yolo.distance_calculator import DistanceCalculator


class Detection:
    def __init__(self, x1, y1, x2, y2, confidence, class_index) -> None:

        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.confidence = confidence
        self.class_index = class_index

    @cached_property
    def bounding_box(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @cached_property
    def width(self):
        return abs(self.x2 - self.x1)

    @cached_property
    def height(self):
        return abs(self.y2 - self.y1)

    @cached_property
    def distance(self):
        return DistanceCalculator.get_object_distance(
            self.x1, self.x2, self.width, self.height
        )

    @staticmethod
    def from_output(output: List):
        return Detection(
            x1=int(output[0]),
            y1=int(output[1]),
            x2=int(output[2]),
            y2=int(output[3]),
            confidence=output[4],
            class_index=int(output[5]),
        )
