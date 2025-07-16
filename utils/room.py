import numpy as np
import cv2
import sys

sys.setrecursionlimit(int(1e8))


class BuildRoom:
    def __init__(self, conf, r, mic_angle):
        self.conf = conf
        self.r = r
        self.mic_angle = mic_angle

    def get_size(self):
        lw = (
            np.random.rand(2) * (self.conf.build.max_lw - self.conf.build.min_lw)
            + self.conf.build.min_lw
        )  # length, width
        h_half = (
            np.random.rand(1) * (self.conf.build.max_h - self.conf.build.min_h)
            + self.conf.build.min_h
        )  # height
        l = lw[0]
        w = lw[1]

        Lnonconvex_x = np.random.rand(1) * ((0.5 * l) - (0)) + (0)
        Lnonconvex_y = np.random.rand(1) * ((0.5 * w) - (0)) + (0)
        Lnonconvex = np.array([Lnonconvex_x, Lnonconvex_y]).squeeze()

        Tnonconvex_x1 = np.random.rand(1) * ((-0.25 * l) - (-0.75 * l)) + (-0.75 * l)
        Tnonconvex_x2 = np.random.rand(1) * ((0.75 * l) - (0.25 * l)) + (0.25 * l)
        Tnonconvex_y = np.random.rand(1) * ((0) - (-0.5 * w)) + (-0.5 * w)
        Tnonconvex = np.array([Tnonconvex_x1, Tnonconvex_x2, Tnonconvex_y]).squeeze()

        return lw, h_half, Lnonconvex, Tnonconvex

    def get_corners(self, lw, Lnonconvex, Tnonconvex, room_name):
        assert len(Lnonconvex) == 2
        assert len(Tnonconvex) == 3
        l = lw[0]
        w = lw[1]

        if room_name == "triangle":
            corners = np.c_[[-l, -w], [l, -w], [-l, w]]  # [x,y]

        elif room_name == "shoebox":
            corners = np.c_[
                [-l, -w],
                [l, -w],
                [l, w],
                [-l, w],
            ]

        elif room_name == "pentagonal":
            corners = np.c_[
                [l * np.cos(2 * np.pi * 1 / 5), w * np.sin(2 * np.pi * 1 / 5)],
                [l * np.cos(2 * np.pi * 2 / 5), w * np.sin(2 * np.pi * 2 / 5)],
                [l * np.cos(2 * np.pi * 3 / 5), w * np.sin(2 * np.pi * 3 / 5)],
                [l * np.cos(2 * np.pi * 4 / 5), w * np.sin(2 * np.pi * 4 / 5)],
                [l * np.cos(2 * np.pi * 5 / 5), w * np.sin(2 * np.pi * 5 / 5)],
            ]

        elif room_name == "hexagonal":
            corners = np.c_[
                [l * np.cos(2 * np.pi * 1 / 6), w * np.sin(2 * np.pi * 1 / 6)],
                [l * np.cos(2 * np.pi * 2 / 6), w * np.sin(2 * np.pi * 2 / 6)],
                [l * np.cos(2 * np.pi * 3 / 6), w * np.sin(2 * np.pi * 3 / 6)],
                [l * np.cos(2 * np.pi * 4 / 6), w * np.sin(2 * np.pi * 4 / 6)],
                [l * np.cos(2 * np.pi * 5 / 6), w * np.sin(2 * np.pi * 5 / 6)],
                [l * np.cos(2 * np.pi * 6 / 6), w * np.sin(2 * np.pi * 6 / 6)],
            ]

        elif room_name == "L":
            corners = np.c_[
                [-l, -w],
                [l, -w],
                [l, Lnonconvex[1]],
                [Lnonconvex[0], Lnonconvex[1]],
                [Lnonconvex[0], w],
                [-l, w],
            ]

        elif room_name == "T":
            corners = np.c_[
                [-l, Tnonconvex[-1]],
                [Tnonconvex[0], Tnonconvex[-1]],
                [Tnonconvex[0], -w],
                [Tnonconvex[1], -w],
                [Tnonconvex[1], Tnonconvex[-1]],
                [l, Tnonconvex[-1]],
                [l, w],
                [-l, w],
            ]

        return corners

    def crushing(self, corners):
        crush = np.random.rand(*corners.shape) * (
            (self.conf.build.crushing_level) - (-self.conf.build.crushing_level)
        ) + (-self.conf.build.crushing_level)
        corners += crush

        return corners

    def rotating(self, corners, theta):
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        corners = np.matmul(rot_matrix, corners)

        return corners

    def get_image_labels(self, corners, resolution, size):
        if_wrong = False
        labels = np.zeros([size, size])

        for i in range(corners.shape[-1]):
            xs, ys = self._get_line_points(corners[:, i - 1], corners[:, i], resolution)
            positions_x = (np.floor(xs * (1 / resolution)) + int(size / 2)).astype(
                np.int64
            )
            positions_y = -(np.floor(ys * (1 / resolution)) + int(size / 2) + 1).astype(
                np.int64
            )
            if sum(abs(positions_x) >= size) > 0 or sum(abs(positions_y) >= size) > 0:
                if_wrong = True  # out-of-bound check
                continue
            labels[positions_y, positions_x] = 1

        ## filled labels
        filled_labels = labels.copy().astype(np.uint8)  # for cv2 lib.
        filled_pixel_num, filled_labels, _, rectangular = cv2.floodFill(
            filled_labels, None, (int(size / 2), int(size / 2)), 1
        )
        filled_labels = ~(filled_labels.astype(bool)) + labels.astype(
            bool
        )  # fill outside of rooms (include walls)
        labels, filled_labels = labels.astype(int), filled_labels.astype(int)

        return labels, filled_labels, if_wrong

    def _get_line_points(self, dot1, dot2, resolution):
        assert dot1[0] != dot2[0]
        dots_x = np.hstack([dot1[0], dot2[0]])
        dots_y = np.hstack([dot1[1], dot2[1]])
        [m, b] = np.polyfit(dots_x, dots_y, 1)

        sample_res = resolution
        if abs(m) < 1:
            xs = np.arange(min(dots_x), max(dots_x) + sample_res, sample_res)
            xs[-1] = max(dots_x)
            ys = (m * xs) + b
        else:
            ys = np.arange(min(dots_y), max(dots_y) + sample_res, sample_res)
            ys[-1] = max(dots_y)
            xs = (ys - b) / m

        return xs, ys

    def get_parametric_labels(self, corners, h_half, src_h, room_name):
        all_corners = self._get_allcorners(corners, h_half, src_h)
        h_src2floor = all_corners[-1, 0]
        h_src2ceiling = all_corners[-1, -1]

        if room_name == "triangle":
            planes = 3
        elif room_name == "shoebox":
            planes = 4
        elif room_name == "pentagonal":
            planes = 5
        elif room_name == "hexagonal":
            planes = 6
        elif room_name == "L":
            planes = 6
        elif room_name == "T":
            planes = 8

        room_model = np.zeros([planes + 2, 3, 3])  # [#planes, #points, xyz]
        labels = np.zeros([self.conf.build.classes, 4])

        for i in range(planes):
            room_model[i] = [
                [all_corners[0, i], all_corners[1, i], all_corners[2, i]],
                [all_corners[0, i + 1], all_corners[1, i + 1], all_corners[2, i + 1]],
                [
                    all_corners[0, i + planes],
                    all_corners[1, i + planes],
                    all_corners[2, i + planes],
                ],
            ]  # 3points on each side
        room_model[-2] = [
            [all_corners[0, 0], all_corners[1, 0], all_corners[2, 0]],
            [all_corners[0, 1], all_corners[1, 1], all_corners[2, 1]],
            [all_corners[0, 2], all_corners[1, 2], all_corners[2, 2]],
        ]  # 3points on floor
        room_model[-1] = [
            [all_corners[0, -1], all_corners[1, -1], all_corners[2, -1]],
            [all_corners[0, -2], all_corners[1, -2], all_corners[2, -2]],
            [all_corners[0, -3], all_corners[1, -3], all_corners[2, -3]],
        ]  # 3points on ceiling

        labels[: planes + 2] = self._normal_coefficients(room_model)
        assert self.conf.build.label_permutation == False  ## Do not permute labels!
        if self.conf.build.label_permutation:
            labels = np.random.permutation(labels)
        walls = (np.sum(abs(labels), axis=1) != 0).astype(np.float32)

        ## Make 'a' of [a b c d] always positive
        assert self.conf.build.label_positive == False
        if self.conf.build.label_positive:
            negative_a = (labels[:, 0] * 1000) < -1
            labels[negative_a] = -labels[negative_a]

        return labels, walls, h_src2floor, h_src2ceiling

    def _get_allcorners(self, corners, h_half, src_h):
        xy = np.hstack([corners, corners])
        z = np.hstack(
            [
                np.repeat(-(h_half + src_h), corners.shape[-1]),
                np.repeat((h_half - src_h), corners.shape[-1]),
            ]
        )
        all_corners = np.vstack([xy, z])
        return all_corners

    def _normal_coefficients(self, rm):
        n = rm.shape[0]
        A, B, C, D = (
            np.zeros([n, 1]),
            np.zeros([n, 1]),
            np.zeros([n, 1]),
            np.zeros([n, 1]),
        )
        for i in range(n):
            A[i] = (
                rm[i, 0, 1] * (rm[i, 1, 2] - rm[i, 2, 2])
                + rm[i, 1, 1] * (rm[i, 2, 2] - rm[i, 0, 2])
                + rm[i, 2, 1] * (rm[i, 0, 2] - rm[i, 1, 2])
            )

            B[i] = (
                rm[i, 0, 2] * (rm[i, 1, 0] - rm[i, 2, 0])
                + rm[i, 1, 2] * (rm[i, 2, 0] - rm[i, 0, 0])
                + rm[i, 2, 2] * (rm[i, 0, 0] - rm[i, 1, 0])
            )

            C[i] = (
                rm[i, 0, 0] * (rm[i, 1, 1] - rm[i, 2, 1])
                + rm[i, 1, 0] * (rm[i, 2, 1] - rm[i, 0, 1])
                + rm[i, 2, 0] * (rm[i, 0, 1] - rm[i, 1, 1])
            )

            D[i] = (
                -rm[i, 0, 0]
                * ((rm[i, 1, 1] * rm[i, 2, 2]) - (rm[i, 2, 1] * rm[i, 1, 2]))
                - rm[i, 1, 0]
                * ((rm[i, 2, 1] * rm[i, 0, 2]) - (rm[i, 0, 1] * rm[i, 2, 2]))
                - rm[i, 2, 0]
                * ((rm[i, 0, 1] * rm[i, 1, 2]) - (rm[i, 1, 1] * rm[i, 0, 2]))
            )

        normal = np.sqrt(np.power(A, 2) + np.power(B, 2) + np.power(C, 2))
        a = A / normal
        b = B / normal
        c = C / normal
        d = D / normal

        coeffs = np.hstack([a, b, c, d])
        return coeffs

    def get_src(self, place_region, ori_region, xmax, xmin, ymax, ymin):
        while True:
            src_x = np.random.rand(1) * (xmax - xmin) + xmin
            src_y = np.random.rand(1) * (ymax - ymin) + ymin
            src = np.array([src_x.squeeze(), src_y.squeeze()])

            if ori_region.is_inside(
                src, include_borders=False
            ) and place_region.is_inside(src, include_borders=False):
                return src

    def get_mic_locs(self, src_pos, optional_r=None):
        """
        check that mic_angle is given as [azimuth, elevation] or [elevation, azimuth]
        this code assumes [azimuth, elevation]
        """
        mic_locs = np.zeros([self.mic_angle.shape[0], 3])
        if optional_r is not None:
            for i in range(mic_locs.shape[0]):
                mic_locs[i] = [
                    optional_r
                    * np.sin(self.mic_angle[i, 1])
                    * np.cos(self.mic_angle[i, 0])
                    + src_pos[0],
                    optional_r
                    * np.sin(self.mic_angle[i, 1])
                    * np.sin(self.mic_angle[i, 0])
                    + src_pos[1],
                    optional_r * np.cos(self.mic_angle[i, 1]) + src_pos[2],
                ]
        else:
            for i in range(mic_locs.shape[0]):
                mic_locs[i] = np.array(
                    [
                        self.r
                        * np.sin(self.mic_angle[i, 1])
                        * np.cos(self.mic_angle[i, 0])
                        + src_pos[0],
                        self.r
                        * np.sin(self.mic_angle[i, 1])
                        * np.sin(self.mic_angle[i, 0])
                        + src_pos[1],
                        self.r * np.cos(self.mic_angle[i, 1]) + src_pos[2],
                    ]
                ).squeeze()

        return mic_locs.T
