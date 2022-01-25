#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import numpy as np
import pygame

from threading import Lock


class YoloDetectionResult:
    def __init__(self, image, conf_threshold, nms_threshold):
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image = image


class YoloClassifier(object):
    def __init__(self, config, weights, classes):
        # config -> filename of config file
        # weights -> filename of weights file
        # classes -> filename of classes file

        self.config = config
        self.weights = weights

        with open(classes, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self._lock = Lock()
        self._net = cv2.dnn.readNet(self.weights, self.config)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.layers_names = self._net.getLayerNames()
        self.output_layers = [
            self.layers_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()
        ]

        np.random.seed(0)
        self._COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, img):
        # https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1 / 255,
            size=(416, 416),
            mean=(0, 0, 0),
            # scalefactor=0.00392,
            # size=(320, 320),
            swapRB=True,
            crop=False,
        )

        with self._lock:
            self._net.setInput(blob)
            outputs = self._net.forward(self.output_layers)

        return blob, outputs

    def get_box_dimensions(self, outputs, height, width, conf_threshold):

        boxes = []
        confs = []
        class_ids = []

        for output in outputs:
            for detect in output:

                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > conf_threshold:

                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)

                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)

        return boxes, confs, class_ids

    def classify(self, image) -> YoloDetectionResult:
        # image must be cv2 image

        height = image.shape[0]
        width = image.shape[1]

        _, outs = self.detect_objects(image)

        result = YoloDetectionResult(image, conf_threshold=0.5, nms_threshold=0.4)

        result.boxes, result.confidences, result.class_ids = self.get_box_dimensions(
            outs, height, width, result.conf_threshold
        )

        return result

    def draw(self, detection: YoloDetectionResult):

        indexes = cv2.dnn.NMSBoxes(
            detection.boxes,
            detection.confidences,
            detection.conf_threshold,
            detection.nms_threshold,
        )

        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(detection.boxes)):
            if i in indexes:

                x, y, w, h = detection.boxes[i]

                label = str(self.classes[detection.class_ids[i]])
                color = self._COLORS[detection.class_ids[i]]

                cv2.rectangle(detection.image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(detection.image, label, (x, y - 5), font, 1, color, 1)

    @staticmethod
    def load_image_file(filename):
        image = cv2.imread(filename)

        return image

    # based on https://gist.github.com/jpanganiban/3844261
    # and https://stackoverflow.com/questions/53101698/how-to-convert-a-pygame-image-to-open-cv-image
    # and https://www.reddit.com/r/pygame/comments/gldeqs/pygamesurfarrayarray3d_to_image_cv2/
    @staticmethod
    def load_image_pygame(surface):

        view = pygame.surfarray.array3d(surface)
        view = view.transpose([1, 0, 2])
        img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        return img_bgr

    @staticmethod
    def image_to_pygame(image):

        im = np.asarray(image)
        surface = pygame.surfarray.make_surface(im)
        return surface
