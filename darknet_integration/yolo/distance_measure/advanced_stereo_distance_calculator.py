import math


class AdvancedStereoDistance:
    @staticmethod
    def distance(
        camera_distance: float, image_width: float, fov: float, x1: float, x2: float
    ) -> float:

        try:
            return AdvancedStereoDistance._distance(
                camera_distance,
                fov,
                fov,
                image_width,
                image_width,
                image_width - x1,
                x2,
            )
        except:
            return -float("inf")

    @staticmethod
    def _distance(A, w1, w2, H1, H2, P1, P2):

        # w1, w2: camera view angle (hFOV)
        # H1, H2: number of horizontal pixels (image width)
        # P1 = W - detection_x
        # P2 = detectoion_x

        w1 = math.radians(w1)
        w2 = math.radians(w2)

        B1 = (math.radians(180) - w1) / 2
        B2 = (math.radians(180) - w2) / 2

        # we dont need to call math.radians for fi and teta
        # since all parameters are already in radians
        fi = P1 * w1 / H1 + B1
        teta = P2 * w2 / H2 + B2
        alpha = math.radians(180) - (teta + fi)

        h = A * math.sin(teta) * math.sin(fi) / math.sin(alpha)

        return round(h, 3)
