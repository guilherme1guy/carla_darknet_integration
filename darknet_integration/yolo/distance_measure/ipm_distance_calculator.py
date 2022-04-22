from functools import cached_property, lru_cache
import math
import numpy as np
import cv2

from .camera_data import CameraData


class IPMDistanceCalculator:

    # u is always column number (ie: 0 to 1280 (in 720p))
    # v is always row number (ie: 0 to 720 (in 720p))

    def __init__(self, camera_data: CameraData) -> None:

        self.camera_data = camera_data

        self.update_properties()

    @staticmethod
    def rotation_matrix(camera_data: CameraData):

        # this is ordered with respect to the unreal coordinate system
        # https://github.com/carla-simulator/carla/issues/2915#issuecomment-744020598

        roll, pitch, yaw = camera_data.rad_rotation

        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)

        RY = np.array(
            [
                [cos_pitch, 0, -sin_pitch],
                [0, 1, 0],
                [sin_pitch, 0, cos_pitch],
            ]
        )

        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)

        RX = np.array(
            [
                [1, 0, 0],
                [0, cos_roll, sin_roll],
                [0, -sin_roll, cos_roll],
            ]
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        RZ = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ]
        )

        return RZ @ RY @ RX

    @staticmethod
    def camera_parameter_matrix(camera_data: CameraData) -> np.ndarray:

        fx = camera_data.fx
        fy = camera_data.fy
        s = camera_data.skew
        u0 = camera_data.center_x
        v0 = camera_data.center_y

        return np.array(
            [
                [fx, s, u0],
                [0, fy, v0],
                [0, 0, 1],
            ]
        )

    def direct_projection(self, world_vec, eps=1e-24):
        """Transforms a point from 'world coordinates' (x_W, y_W, z_W) [m] -> 'image coordinates' (x_I, y_I) [px]

        Args:
            world_vec: Column vector (3,1) [m]
            P: Rotation matrix (world -> image coordinates)
            t: Translation vector (world -> image coordinates)

        Returns:
            Image coordinate vector representing (x_I, y_I) pixel location of world coordinates (x_W, y_W, z_W)
        """
        P, t = self.projection_matrix()
        img_vec = P @ world_vec + t
        img_vec = img_vec[:2, :] / (img_vec[2, :] + eps)

        return img_vec

    def inverse_projection(self, img_x, img_y):
        """Transforms a point from 'image coordinates' (x_I, y_I) [px] -> 'world (plane) coordinates' (x_W, y_W, z_W) [m] where z = 0

        Args:
            img_x: Image 'x' coordinate [px]
            img_y: Image 'y' coordinate [px]
            P: Rotation matrix (world -> image coordinates)
            t: Translation vector (world -> image coordinates)

        Returns:
            World coordinate vector (x_W, y_W, z_W) [m] representing road plane location of image coordinate (x_I, y_I)
        """

        P, t = self.projection_matrix()

        # Inverted matrix
        A = np.zeros((4, 4))
        A[0:3, 0:3] = P
        A[0, 3] = -img_x
        A[1, 3] = -img_y
        A[2, 3] = -1
        A[3, 2] = 1

        A_inv = np.linalg.inv(A)

        # Column vector
        t_vec = np.zeros((4, 1))
        t_vec[0:3, :] = -t

        world_coord = A_inv @ t_vec

        return world_coord[:3]

    def update_properties(self):
        self.projection_matrix.cache_clear()
        self.build_img_vecs.cache_clear()
        self.max_distance.cache_clear()

    @lru_cache(maxsize=1)
    def projection_matrix(self):
        # Intrinsic parameter matrix
        K = self.camera_parameter_matrix(self.camera_data)

        # Camera -> Road transformation (given)
        R_cam2road = self.rotation_matrix(self.camera_data)
        T_cam2road = np.transpose(
            np.array([self.camera_data.translation])
            # + np.array(self.camera_data.camera_distance) / 2
        )

        # Road -> Camera transformation (wanted)
        R_road2cam = np.transpose(R_cam2road)
        T_road2cam = -T_cam2road

        # Camera -> Image transformation added to the 'K' matrix
        # NOTE: The cam2img rotation matrix is inductively derived to perform the desired transformation
        #       Camera frame       Image frame
        #           x_cam     -->    -y_img
        #           y_cam     -->    -z_img
        #           z_cam     -->     x_img
        #
        #       {v_cam}^T R_cam2img = {v_img}^T
        #
        R_cam2img = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        C = K @ R_cam2img

        # Compute 'P' matrix and 't' vector mapping 'World coordinate Q' -> 'Image coordinate q_h'
        P = C @ R_road2cam
        t = C @ T_road2cam

        return P, t

    @lru_cache(maxsize=1)
    def max_distance(self):

        mid_point = self.inverse_projection(
            self.camera_data.image_width / 2, self.camera_data.image_height * 0.55
        ).flatten()

        return abs(mid_point[0])

    @lru_cache(maxsize=1)
    def build_img_vecs(self):

        max_distance = self.max_distance()
        x_min = -int(max_distance)
        x_max = int(max_distance)
        y_min = -int(max_distance)
        y_max = int(max_distance)

        x_N = int((x_max - x_min) * 10)
        y_N = int((y_max - y_min) * 10)

        # Construct a matrix by concatenating column vectors, each representing a point in world coordinates
        xs, ys = np.meshgrid(
            np.linspace(x_min, x_max, x_N), np.linspace(y_min, y_max, y_N)
        )

        xs = xs.flatten()
        ys = ys.flatten()
        zs = np.zeros(xs.shape)

        world_vecs = np.array([xs, ys, zs])

        # Project all 'world coordinate' vectors (X, Y, Z) into 'image coordinate' vectors (i, j) at once
        img_vecs = self.direct_projection(world_vecs)

        return np.reshape(img_vecs, (2, x_N, y_N))

    def project(self, img):

        max_point = np.transpose(np.array([[self.max_distance(), 0.0, 0.0]]))
        img_vec = self.direct_projection(max_point)
        img_y_min_limit = int(np.round(img_vec[1, 0]))

        img_vecs = self.build_img_vecs()
        x_N = img_vecs.shape[1]
        y_N = img_vecs.shape[2]

        # Draw the top-down image representation "pixel-by-pixel"
        bev = np.zeros((x_N, y_N, 3), dtype=np.uint8)

        IMG_W = self.camera_data.image_width
        IMG_H = self.camera_data.image_height

        for i in range(x_N):
            for j in range(y_N):

                # For each pixel (i, j), the correponding location in the image frame is obtained from the previous direct mapping operation
                x_I = int(img_vecs[0, i, j])
                y_I = int(img_vecs[1, i, j])

                # Only map pixels within the given image frame
                # 'img_y_min_limit' correspond to range limit 50m
                if 0 <= x_I < IMG_W and img_y_min_limit < y_I < IMG_H:

                    # Map image frame pixel to world coordinate pixel
                    try:
                        bev[y_N - j, x_N - i] = img[y_I, x_I]
                    except:
                        continue

        return self.crop(bev)

    @staticmethod
    def crop(img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            crop = img[y : y + h, x : x + w]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                return img

            return crop
        except:
            return img

    @staticmethod
    def distance_from_points(x, y):
        return round(math.sqrt(x ** 2 + y ** 2), 3)
