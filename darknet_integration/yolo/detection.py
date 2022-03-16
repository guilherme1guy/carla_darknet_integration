from functools import cached_property
from typing import List, Optional

from yolo.distance_measure.simple_distance_calculator import (
    DistanceCalculator,
)


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

    @cached_property
    def distance_pivot(self):
        # pivot is at the botton of the bounding box
        return ((self.x1 + self.x2) / 2, self.y1)

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

    def is_similar(self, other_detection: "Detection", max_error: float = 1):

        if Detection.minimum_mean_square_error(self, other_detection, 0.7) > max_error:
            return False

        return True

    def most_similar(
        self, detections: List["Detection"], ponder: float = 0.7
    ) -> "Detection":

        errors = [
            (Detection.minimum_mean_square_error(self, other, ponder), other)
            for other in detections
        ]
        min_error = min(errors, key=lambda x: x[0])

        return min_error[1]

    @staticmethod
    def minimum_mean_square_error(
        detection: "Detection", other: "Detection", ponder: float
    ) -> float:

        a = (detection.width - other.width) ** 2
        b = (detection.height - other.height) ** 2
        mmse = ponder * a + (1 - ponder) * b

        return mmse

    def __str__(self):
        return f"Detection(\n\tx1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, width={self.width}, height={self.height}, \n\tconfidence={self.confidence}, class_index={self.class_index},\n\t distance_pivot={self.distance_pivot}, \n\tipm_distance={self.ipm_distance}, ipm_x={self.ipm_x}, ipm_z={self.ipm_z}, \n\tsimple_distance={self.simple_distance})"
