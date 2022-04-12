import random

import carla

from game.camera_manager import CameraManager
from game.camera_parser import CameraParser
from game.transform_data import TransformData
from game.world import World
from local_utils import get_actor_blueprints

import threading
import time
import os
import shutil

lock = threading.Lock()


class DatasetGenerator:

    Ys = [
        (-132.75, "3m"),  # 3m
        (-131.75, "4m"),
        (-130.75, "5m"),
        (-129.75, "6m"),
        (-128.75, "7m"),
        (-127.75, "8m"),
        (-126.75, "9m"),
        (-125.75, "10m"),
        (-124.75, "11m"),
        (-123.75, "12m"),
        (-122.75, "13m"),
        (-121.75, "14m"),
        (-120.75, "15m"),
        (-119.75, "16m"),
        (-118.75, "17m"),
        (-117.75, "18m"),
        (-116.75, "19m"),
        (-115.75, "20m"),
        (-114.75, "21m"),
        (-113.75, "22m"),
        (-112.75, "23m"),
        (-111.75, "24m"),
        (-110.75, "25m"),  # 25m
    ]

    OUT_PATH = "_out"
    DATASET_PATH = "dataset"

    def __init__(self, world: World) -> None:
        self.world: World = world
        self.camera_manager: CameraManager = world.camera_manager
        self.camera_parser = world.camera_manager.camera_parser

        self.create_local_subdirectories(self.OUT_PATH)
        self.create_local_subdirectories(self.DATASET_PATH)

    def create_local_subdirectories(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def generate_dataset(self):
        t = threading.Thread(target=self._generate_dataset)
        self.clear_dataset_path()
        t.start()

    def _generate_dataset(self):

        lock.acquire()

        print("[dgen] Generating dataset...")
        print("[dgen] Setting player...")

        player: carla.Vehicle = self.world.player

        player_position = TransformData(385, -140, 0, yaw=90)
        player.set_transform(player_position.transform)

        print("[dgen] Setting other...")

        other = self.create_other_vehicle(
            TransformData(385, self.Ys[0][0], 0.5, yaw=90).transform
        )

        print("[dgen] Starting position phase...")
        for y_position, name in self.Ys:

            self.camera_parser.recording = False

            print(f"[dgen] Setting other position to {y_position} ({name})...")

            transform_data = TransformData(385, y_position, 0, yaw=90)
            other.set_transform(transform_data.transform)

            print("[dgen] Waiting for other to reach position...")
            while not self.camera_parser.yolov3.jobs.empty():
                self.camera_parser.yolov3.jobs.get(timeout=0.1)
            while not self.camera_parser.yolov5.jobs.empty():
                self.camera_parser.yolov5.jobs.get(timeout=0.1)
            time.sleep(7)

            print("[dgen] Starting recording...")
            self.clear_out_path()
            self.camera_parser.recording = True

            while len(os.listdir(self.OUT_PATH)) < 8:
                time.sleep(1)

            self.camera_parser.recording = False

            print("[dgen] Copying files...")
            files = list(reversed(sorted(os.listdir(self.OUT_PATH))))
            bases = list(
                reversed(
                    sorted(
                        list(
                            set(
                                [
                                    filename.replace("_1.png", "").replace("_0.png", "")
                                    for filename in files
                                ]
                            )
                        )
                    )
                )
            )

            first = True
            for base in bases:
                if f"{base}_0.png" in files and f"{base}_1.png" in files:
                    if first:
                        first = False
                        continue
                    shutil.copy(
                        f"{self.OUT_PATH}/{base}_0.png",
                        f"{self.DATASET_PATH}/{name}_0.png",
                    )
                    print(
                        f"[dgen] Copied {self.OUT_PATH}/{base}_0.png -> {self.DATASET_PATH}/{name}_0.png"
                    )
                    shutil.copy(
                        f"{self.OUT_PATH}/{base}_1.png",
                        f"{self.DATASET_PATH}/{name}_1.png",
                    )
                    print(
                        f"[dgen] Copied {self.OUT_PATH}/{base}_1.png -> {self.DATASET_PATH}/{name}_1.png"
                    )
                    break

            print("[dgen] Removing files...")
            # delete old files
            self.clear_out_path()

            print(f"[dgen] Finished ({name})...")
            # wait before next image, so the render queue can update
            time.sleep(1)

        print("[dgen] Removing other vehicle...")
        other.destroy()

        print("[dgen] Dataset generated!")
        lock.release()

    def clear_out_path(self):
        for filename in reversed(sorted(os.listdir(self.OUT_PATH))):
            os.remove(f"{self.OUT_PATH}/{filename}")

    def clear_dataset_path(self):
        for filename in reversed(sorted(os.listdir(self.DATASET_PATH))):
            os.remove(f"{self.DATASET_PATH}/{filename}")

    def create_other_vehicle(self, spawn_point):
        blueprint = random.choice(
            get_actor_blueprints(self.world.world, "vehicle.tesla.model3", "2")
        )
        blueprint.set_attribute("role_name", "other")
        if blueprint.has_attribute("color"):
            blueprint.set_attribute(
                "color", blueprint.get_attribute("color").recommended_values[0]
            )

        if blueprint.has_attribute("driver_id"):
            blueprint.set_attribute(
                "driver_id", blueprint.get_attribute("driver_id").recommended_values[0]
            )

        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")

        # Spawn the vehicle.
        vehicle = self.world.world.spawn_actor(blueprint, spawn_point)

        return vehicle
