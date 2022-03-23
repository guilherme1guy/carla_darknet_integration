from functools import cached_property, lru_cache
import math
import numpy as np

from .camera_data import CameraData


class IPMDistanceCalculator:

    # u is always column number (ie: 0 to 1280 (in 720p))
    # v is always row number (ie: 0 to 720 (in 720p))

    def __init__(self, camera_data: CameraData) -> None:

        self.camera_data = camera_data
        self.max_distance = 50.0

        self.update_properties()

    @staticmethod
    def rotation_matrix(camera_data: CameraData):

        roll, pitch, yaw = camera_data.rad_rotation

        si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
        ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        R = np.identity(3)

        R[0, 0] = cj * ck
        R[0, 1] = sj * sc - cs
        R[0, 2] = sj * cc + ss

        R[1, 0] = cj * sk
        R[1, 1] = sj * ss + cc
        R[1, 2] = sj * cs - sc

        R[2, 0] = -sj
        R[2, 1] = cj * si
        R[2, 2] = cj * ci

        return R

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

    @lru_cache(maxsize=1)
    def projection_matrix(self):
        # Intrinsic parameter matrix
        K = self.camera_parameter_matrix(self.camera_data)

        # Camera -> Road transformation (given)
        R_cam2road = self.rotation_matrix(self.camera_data)
        T_cam2road = np.array([self.camera_data.translation]).T

        # Road -> Camera transformation (wanted)
        R_road2cam = R_cam2road.T
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
    def build_img_vecs(self):

        x_min = -int(self.max_distance)
        x_max = int(self.max_distance)
        y_min = -int(self.max_distance)
        y_max = int(self.max_distance)

        res = 0.05

        x_N = int((x_max - x_min) / res)
        y_N = int((y_max - y_min) / res)

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

        max_point = np.array([[self.max_distance, 0.0, 0.0]]).T
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

        return bev

    @staticmethod
    def distance_from_points(x, y):
        return round(math.sqrt(x ** 2 + y ** 2), 2)
