import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
import torch

torch.set_num_threads(1)
from tqdm import tqdm
import time


def main(conf):
    data_path = conf["data_path"]
    dataset_name = conf["dataset_name"]
    misc_name = conf["misc_name"]
    train_valid_test = train_valid_test_check(data_path)
    data = np.load(data_path, allow_pickle=True)

    for iter in tqdm(range(conf["iterations"][0], conf["iterations"][1])):
        for iter_room, room_xyz in enumerate(tqdm(data)):
            room_xyz = np.array(room_xyz)
            ori_corners3d = room_xyz.T.copy()

            ori_total_height = abs(ori_corners3d[-1, -1] - ori_corners3d[-1, 0])
            if ori_total_height > 6.0:  # maximum
                ori_total_height = 6.0
            elif ori_total_height < 2.0:  # minimum
                ori_total_height = 2.0

            ori_corners2d = ori_corners3d[:2, : (ori_corners3d.shape[-1] // 2)].copy()
            if np.array_equal(
                ori_corners2d[:, : (ori_corners2d.shape[-1] // 2)],
                ori_corners2d[:, (ori_corners2d.shape[-1] // 2) :],
            ):  # fix double-corner error
                ori_corners2d = ori_corners2d[:, : (ori_corners2d.shape[-1] // 2)]
            ori_corners2d = set_corners2d_to_middle(ori_corners2d)

            if train_valid_test == "train":
                conf["ori_room_size_variation"] = get_random_in_range(0.5, 2.0)
                ori_corners2d = ori_corners2d * conf["ori_room_size_variation"]
                while True:
                    ori_total_height_variated = ori_total_height * get_random_in_range(
                        0.5, 2.0
                    )
                    if (
                        ori_total_height_variated > 2.0
                        and ori_total_height_variated < 6.0
                    ):
                        break
                ori_total_height = ori_total_height_variated
            pr_corners2d = ori_corners2d * conf["place_region_rate"]

            src_xy, if_wrong_src = get_src(ori_corners2d, pr_corners2d)  # [2,]
            if if_wrong_src:
                continue

            ## floor boundary ##
            parallel_trans_floor_boundary = np.expand_dims(src_xy.copy(), -1)
            floor_boundary = ori_corners2d - parallel_trans_floor_boundary
            if conf["rotation"]:
                theta = (np.random.rand(1) * (2 * np.pi)).squeeze()
                floor_boundary = rotation(floor_boundary, theta)
            filled_image_label, if_wrong_image_label = get_image_label(
                corners=floor_boundary,
                resolution=conf["resolution"],
                size=conf["image_label_size"],
            )

            ## floor ceiling ##
            src_z = get_random_in_range(
                1, 1.5
            )  # source is located at 1~1.5m from floor
            h_src2floor = -(src_z.copy())
            h_src2ceiling = ori_total_height - src_z
            floor_ceiling = np.hstack([h_src2floor, h_src2ceiling])
            filled_height_label, if_wrong_height_label = get_height_label(
                h_src2floor=h_src2floor,
                h_src2ceiling=h_src2ceiling,
                resolution=conf["resolution"],
                size=conf["height_label_size"],
            )
            if if_wrong_image_label or if_wrong_height_label:
                continue

            ## get Room ##
            room = get_pyroom_room(conf, floor_boundary, floor_ceiling)
            rir = get_pyroom_rir(conf, room)
            rt60 = room.measure_rt60().mean()

            ## packing ##
            rir = torch.Tensor(rir)
            filled_image_label = torch.Tensor(filled_image_label)
            height_labels = torch.Tensor(filled_height_label)
            rt60 = torch.Tensor([rt60])

            os.makedirs(dataset_name, exist_ok=True)

            torch.save(
                {
                    "rir": rir,
                    "filled_image_labels": filled_image_label,
                    "height_labels": height_labels,
                    "rt60": rt60,
                },
                f"{dataset_name}/{misc_name}_{iter}_{iter_room}.pt",
            )


def get_pyroom_room(conf, floor_boundary, floor_ceiling):
    parallel_trans_pyroom = np.array(
        [[-min(floor_boundary[0, :]), -min(floor_boundary[1, :])]]
    ).T  # Pyroomacoustics can't calculate RIR when vertex has minus
    floor_boundary_pyroom = floor_boundary + parallel_trans_pyroom
    src_loc = parallel_trans_pyroom.copy()
    src_loc = np.vstack([src_loc, abs(floor_ceiling[0])])
    mic_locs = get_mic_locs(src_loc, conf["device_radius"], conf["mic_arr_channel"])

    ## build room ##
    height = sum(abs(floor_ceiling))
    assert (
        conf["use_random_abs"] + conf["use_materials"] == 1
    )  # select one of [use_randomabs, use_materials]

    if conf["use_random_abs"]:
        assert len(conf["abs_coeff"]) == 2
        abs_coeff = (
            get_random_in_range(conf["abs_coeff"][0], conf["abs_coeff"][1])
            .squeeze()
            .item()
        )

        abs_M = pra.Material(abs_coeff)
        abs_M_flo_ceil = pra.Material(abs_coeff)
    else:
        abs_M = pra.Material(energy_absorption=random.choice(conf["abs_material"]))
        abs_M_flo_ceil = pra.make_materials(
            floor=random.choice(conf["abs_material_floor"]),
            ceiling=random.choice(conf["abs_material_ceiling"]),
        )

    if conf["use_random_abs"]:
        room = pra.Room.from_corners(
            floor_boundary_pyroom,
            fs=conf["fs"],
            max_order=conf["max_order"],
            materials=abs_M,
            air_absorption=False,
            ray_tracing=True,
        )
        room.extrude(
            height=height,
            materials=abs_M_flo_ceil,
        )
    elif conf["use_materials"]:
        room = pra.Room.from_corners(
            floor_boundary_pyroom,
            fs=conf["fs"],
            max_order=conf["max_order"],
            materials=abs_M,
            air_absorption=False,
            ray_tracing=True,
        )
        room.extrude(
            height=height,
            materials=abs_M_flo_ceil,
        )

    if conf["simulation_type"] == "ism":
        pass
    elif conf["simulation_type"] == "ray":
        room.set_ray_tracing(
            n_rays=conf["n_rays"], receiver_radius=conf["receiver_radius"]
        )
    else:
        raise Exception("Incorrect simulation type")

    room.add_source(src_loc)
    room.add_microphone_array(mic_locs)

    return room


def get_mic_locs(src_pos, radius, mic_arr_channel=6):
    src_x, src_y, src_z = (
        src_pos[0].squeeze().copy(),
        src_pos[1].squeeze().copy(),
        src_pos[2].squeeze().copy(),
    )

    if mic_arr_channel == 1:
        mic_locs = np.expand_dims(src_pos.copy(), 0)
        return mic_locs
    elif mic_arr_channel == 4:
        mic_angle = np.array(
            [
                [1 * np.pi / 2, np.pi / 2],
                [2 * np.pi / 2, np.pi / 2],
                [3 * np.pi / 2, np.pi / 2],
                [4 * np.pi / 2, np.pi / 2],
            ]
        )
    elif mic_arr_channel == 6:
        mic_angle = np.array(
            [
                [3 * np.pi / 3, np.pi / 2],
                [4 * np.pi / 3, np.pi / 2],
                [5 * np.pi / 3, np.pi / 2],
                [6 * np.pi / 3, np.pi / 2],
                [1 * np.pi / 3, np.pi / 2],
                [2 * np.pi / 3, np.pi / 2],
            ]
        )
    else:
        raise ValueError("wrong mic_arr_channel")

    mic_locs = np.zeros([mic_angle.shape[0], 3])
    for i in range(mic_locs.shape[0]):
        mic_locs[i] = [
            radius * np.sin(mic_angle[i, 1]) * np.cos(mic_angle[i, 0]) + src_x,
            radius * np.sin(mic_angle[i, 1]) * np.sin(mic_angle[i, 0]) + src_y,
            radius * np.cos(mic_angle[i, 1]) + src_z,
        ]

    return mic_locs.T


def get_image_label(corners, resolution, size):
    """
    label: 1 for floor-boundary-edge
    filled_label: 0 for inside of floor-boundary, 1 for outside of floor-boundary
    """
    assert len(corners.shape) == 2
    if_wrong = False
    label = np.zeros([size, size])
    limit = int(size / 2 * resolution)

    if (
        abs(np.min(corners)) > limit or abs(np.max(corners)) > limit
    ):  # checking out-of-bound
        if_wrong = True
        return None, if_wrong

    for i in range(corners.shape[-1]):
        xs, ys = _get_line_points(corners[:, i - 1], corners[:, i], resolution)
        positions_x = (np.floor(xs * (1 / resolution)) + int(size / 2)).astype(np.int64)
        positions_y = -(np.floor(ys * (1 / resolution)) + int(size / 2) + 1).astype(
            np.int64
        )
        if (
            sum(abs(positions_x) >= size) > 0 or sum(abs(positions_y) >= size) > 0
        ):  # checking out-of-bound
            if_wrong = True
            return None, if_wrong
        else:
            label[positions_y, positions_x] = 1

    ## filled label
    filled_label = label.copy().astype(np.uint8)  # for cv2 lib.
    filled_pixel_num, filled_label, _, rectangular = cv2.floodFill(
        filled_label, None, (int(size / 2), int(size / 2)), 1
    )
    filled_label = ~(filled_label.astype(bool)) + label.astype(
        bool
    )  # fill outside of rooms (include walls)
    label, filled_label = label.astype(int), filled_label.astype(int)
    if np.sum(filled_label) == 0 or np.sum(filled_label) == int(size * size):
        if_wrong = True
        return None, if_wrong

    return filled_label, if_wrong


def get_height_label(h_src2floor, h_src2ceiling, resolution, size):
    if_wrong = False
    half_of_size = int(size / 2)

    filled_label = np.ones(size)
    filled_floor = int(np.round(abs(h_src2floor) / resolution))
    filled_ceiling = int(np.round(abs(h_src2ceiling) / resolution))

    if (
        filled_floor > half_of_size or filled_ceiling > half_of_size
    ):  # checking out-of-bound
        if_wrong = True
        return None, if_wrong

    filled_label[half_of_size : (half_of_size + filled_floor)] = 0  # floor
    filled_label[(half_of_size - filled_ceiling) : half_of_size] = 0  # ceiling
    filled_label = filled_label.astype(int)

    return filled_label, if_wrong


def _get_line_points(dot1, dot2, resolution):
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


def get_src(ori_corners2d, pr_corners2d, maximum_finding=100):
    if_wrong = False

    pr_xmin, pr_xmax = min(pr_corners2d[0, :]), max(pr_corners2d[0, :])
    pr_ymin, pr_ymax = min(pr_corners2d[1, :]), max(pr_corners2d[1, :])

    place_region = pra.Room.from_corners(pr_corners2d)
    ori_region = pra.Room.from_corners(ori_corners2d)

    for _ in range(maximum_finding):
        src_x = get_random_in_range(pr_xmin, pr_xmax)
        src_y = get_random_in_range(pr_ymin, pr_ymax)
        src = np.array([src_x.squeeze(), src_y.squeeze()])

        if ori_region.is_inside(src, include_borders=False) and place_region.is_inside(
            src, include_borders=False
        ):
            return src, if_wrong
    if_wrong = True

    return None, if_wrong


def set_corners2d_to_middle(corners2d):
    parallel_trans_x = (min(corners2d[0, :]) + max(corners2d[0, :])) / 2
    parallel_trans_y = (min(corners2d[1, :]) + max(corners2d[1, :])) / 2
    parallel_trans = np.vstack([parallel_trans_x, parallel_trans_y])

    corners2d_t = corners2d - parallel_trans

    return corners2d_t


def get_random_in_range(min_val, max_val, nb_randoms=1):
    return np.random.rand(nb_randoms) * (max_val - min_val) + min_val


def rotation(corners, theta):
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    corners = np.matmul(rot_matrix, corners)

    return corners


def get_pyroom_rir(conf, room):
    room.compute_rir()
    if conf["remove_direct_peak"]:
        eps_sample = int(0.002 * conf["fs"])  # 2ms after direct peak

        temp_rir_len = conf["fs"]
        temp_rir = np.zeros([conf["mic_arr_channel"], temp_rir_len])
        for i in range(conf["mic_arr_channel"]):
            if len(room.rir[i][0]) < temp_rir_len:
                temp_rir[i, : len(room.rir[i][0])] = room.rir[i][0]
            else:
                temp_rir[i] = room.rir[i][0][:temp_rir_len]

        ## normalize: direct peak to 1
        temp_rir = temp_rir - np.mean(temp_rir, axis=-1, keepdims=True)
        temp_rir = temp_rir / (np.max(np.abs(temp_rir), axis=-1, keepdims=True) + 1e-8)

        ## remove direct
        direct_peak_idx = np.argmax(np.abs(temp_rir)[0])
        direct_peak_idx += eps_sample
        rir = temp_rir[:, direct_peak_idx : conf["rir_length"] + direct_peak_idx]
    else:
        raise NotImplementedError("We trimed direct part of RIR")

    return rir


def train_valid_test_check(data_path):
    if "train" in data_path:
        train_valid_test = "train"
    elif "val" in data_path:
        train_valid_test = "valid"
    elif "test" in data_path:
        train_valid_test = "test"

    return train_valid_test


if __name__ == "__main__":
    conf = {
        "data_path": "data/manhattan_train.npy",
        "dataset_name": "dataset/manhattan_train/",
        "misc_name": "manhattan",
        # 'misc_name': 'atlanta',
        ## basic settings ##
        "iterations": [0, 1],
        "save_on": True,
        "waiting_for_save": False,
        "device_radius": 0.05,
        "image_label_size": 1024,
        "height_label_size": 512,
        "resolution": 0.02,
        "place_region_rate": 0.7,
        "rotation": True,
        "remove_direct_peak": True,
        ## pyroom settings ##
        "simulation_type": "ray",  # ['ism', 'ray']
        "max_order": 6,
        "n_rays": 10000,
        "receiver_radius": 0.5,
        "use_random_abs": False,
        "use_materials": True,
        "abs_coeff": [0.1, 0.3],
        "abs_material": [
            "hard_surface",
            "glass_window",
            "rough_concrete",
            "rough_lime_wash",
            "plasterboard",
        ],
        "abs_material_floor": ["linoleum_on_concrete", "carpet_thin", "audience_floor"],
        "abs_material_ceiling": [
            "ceiling_perforated_gypsum_board",
            "ceiling_metal_panel",
            "ceiling_plasterboard",
        ],
        ## default settings ##
        "mic_arr_channel": 6,
        "rir_length": 1024,
        "fs": 8000,
    }

    main(conf)
