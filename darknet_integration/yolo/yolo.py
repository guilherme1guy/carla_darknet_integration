#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import numpy as np
import pygame


class YoloDetectionResult:
    def __init__(self, image, conf_threshold, nms_threshold):
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.indices = []
        self.image = image


class YoloClassifier(object):
    def __init__(self, config, weights, classes):
        # config -> filename of config file
        # weights -> filename of weights file
        # classes -> filename of classes file

        self.config = config
        self.weights = weights
        self.classes = classes

        self._net = cv2.dnn.readNet(self.weights, self.config)
        self._COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def classify(self, image) -> YoloDetectionResult:
        # image must be cv2 image

        width = image.shape[1]
        height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False
        )

        self._net.setInput(blob)

        outs = self._net.forward(self.get_output_layers(self._net))

        result = YoloDetectionResult(image, conf_threshold=0.5, nms_threshold=0.4)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    result.class_ids.append(class_id)
                    result.confidences.append(float(confidence))
                    result.boxes.append([x, y, w, h])

        result.indices = cv2.dnn.NMSBoxes(
            result.boxes,
            result.confidences,
            result.conf_threshold,
            result.nms_threshold,
        )

        return result

    def draw(self, detection: YoloDetectionResult):
        # image must be a cv2 image

        for i in detection.indices:
            i = i[0]
            box = detection.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self.draw_prediction(
                detection.image,
                detection.class_ids[i],
                detection.confidences[i],
                round(x),
                round(y),
                round(x + w),
                round(y + h),
            )

        # cv2.imshow("object detection", image)
        # cv2.waitKey()

        # cv2.imwrite("object-detection.jpg", image)
        # cv2.destroyAllWindows()

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # img must be a cv2 image

        label = str(self.classes[class_id])

        color = self._COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        return img

    @staticmethod
    def get_output_layers(net):

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

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
    def cvimage_to_pygame(image):

        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = np.rot90(np.fliplr(im))
        surface = pygame.surfarray.make_surface(im)

        return surface
