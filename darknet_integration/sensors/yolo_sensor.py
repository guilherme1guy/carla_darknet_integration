import collections
import io
import math
import weakref

import carla
import cv2
import json
import numpy as np

from darknet_integration.yolo import YoloClassifier, YoloDetectionResult


class YoloSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud

        world = self._parent.get_world()

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "1280")
        blueprint.set_attribute("image_size_y", "720")
        blueprint.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=0.8, z=2))

        self.sensor = world.spawn_actor(blueprint, transform, attach_to=self._parent)

        self.last_result = None

        self.yolo = YoloClassifier(
            config="/home/gguy/code/darknet/cfg/yolov3.cfg",
            weights="/home/gguy/code/darknet/yolov3.weights",
            classes="/home/gguy/code/darknet/data/coco.names",
        )

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: YoloSensor._on_tick(weak_self, data))

    @staticmethod
    def _on_tick(weak_self, data):

        self = weak_self()
        if not self:
            return

        # https://github.com/carla-simulator/carla/blob/d23f3dc1340e47265eeea2b1b33b2d3a2d6d4f42/PythonAPI/examples/visualize_multiple_sensors.py#L170
        data.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        # array = array.transpose([1, 0, 2])

        yolo: YoloClassifier = self.yolo
        result: YoloDetectionResult = yolo.classify(
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        )

        yolo.draw(result)

        self.last_result = result

        # cv2.imwrite(f"_out/{data.timestamp}.jpg", result.image)

        # j_data = {
        #     "class_ids": result.class_ids,
        #    "confidences": result.confidences,
        #    "boxes": result.boxes,
        #    "conf_threshold": result.conf_threshold,
        #    "nms_threshold": result.nms_threshold,
        # }

        # print(j_data)

        # with open(f"_out/{data.timestamp}.json", "w") as file:
        #    file.write(json.dumps(j_data))
