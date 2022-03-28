import math
from typing import List, Optional

import carla
import numpy as np
from yolo.distance_measure.ipm_distance_calculator import IPMDistanceCalculator

DRAW_DEBUG = False


class GroundTruthDistance:
    def __init__(self, ego: carla.Vehicle) -> None:

        self.ego = ego
        self.ego_tansform = ego.get_transform()

        self.ego_points = GroundTruthDistance.get_measuring_points(ego)

    def simple_distance(self, other: carla.Vehicle) -> float:
        return self.ego_tansform.location.distance(other.get_transform().location)

    def distance(self, other: carla.Vehicle):

        other_points = GroundTruthDistance.get_measuring_points(other)

        dist_func = lambda a, b: math.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        )
        distances = [
            (dist_func(other_point, ego_point), other_point, ego_point)
            for other_point in other_points
            for ego_point in self.ego_points
        ]

        return min(distances, key=lambda x: x[0])

    @staticmethod
    def draw_point(world, point: carla.Vector3D):
        if not DRAW_DEBUG:
            return

        world.debug.draw_point(
            point,
            size=0.1,
            color=carla.Color(0, 0, 255),
            life_time=0.1,
        )

    @staticmethod
    def draw_points(world, points: List[carla.Vector3D]):
        if not DRAW_DEBUG:
            return

        for point in points:
            GroundTruthDistance.draw_point(world, point)

    @staticmethod
    def draw_line(world, a: carla.Vector3D, b: carla.Vector3D):

        if not DRAW_DEBUG:
            return

        world.debug.draw_line(
            carla.Location(a.x, a.y, a.z),
            carla.Location(b.x, b.y, b.z),
            thickness=0.01,
            life_time=0.1,
            color=carla.Color(r=0, g=255, b=0, a=50),
            persistent_lines=False,
        )

    @staticmethod
    def draw_bounding_box(world, v: carla.Vehicle):

        if not DRAW_DEBUG:
            return

        transform: carla.Transform = v.get_transform()

        box: carla.BoundingBox = v.bounding_box
        box.location += transform.location

        world.debug.draw_box(
            box,
            transform.rotation,
            thickness=0.01,
            life_time=0.1,
            color=carla.Color(r=0, g=255, b=0, a=50),
        )

    @staticmethod
    def get_measuring_points(
        v: carla.Vehicle, transform: Optional[carla.Transform] = None
    ):

        if transform is None:
            transform = v.get_transform()

        extent = v.bounding_box.extent
        box_center = v.bounding_box.location
        transform_center = transform.location
        center = box_center + transform_center

        displacements = [
            # corners
            carla.Location(extent.x, extent.y, -extent.z),
            carla.Location(-extent.x, extent.y, -extent.z),
            carla.Location(extent.x, -extent.y, -extent.z),
            carla.Location(-extent.x, -extent.y, -extent.z),
            # midpoints
            carla.Location(extent.x, 0, -extent.z),
            carla.Location(-extent.x, 0, -extent.z),
            carla.Location(0, extent.y, -extent.z),
            carla.Location(0, -extent.y, -extent.z),
        ]

        points = []

        # obtain rotation matrix
        # R = IPMDistanceCalculator._rotation_matrix(
        #     transform.rotation.roll,
        #     transform.rotation.pitch,
        #     transform.rotation.yaw,
        # )

        R = np.array(transform.get_inverse_matrix())
        R = R[:3, :3]

        for displacement in displacements:

            # apply 3d rotation
            displacement_coordinates = (
                np.array([displacement.x, displacement.y, displacement.z]) @ R
            )

            displacement.x = displacement_coordinates[0]
            displacement.y = displacement_coordinates[1]
            displacement.z = displacement_coordinates[2]

            point = center + displacement
            points.append(point)

        return points
