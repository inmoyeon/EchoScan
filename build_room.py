from configwrapper import ConfigWrapper
from utils.room import BuildRoom

import os
import numpy as np
from tqdm import tqdm
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import torch

torch.set_num_threads(1)
import random
import matplotlib.pyplot as plt


def main(repeat_room):
    room_type = np.random.choice(
        conf.build.room_types, size=1, p=conf.build.room_prob
    ).squeeze()
    if room_type == 3:
        room_name = "triangle"
    elif room_type == 4:
        room_name = "shoebox"
    elif room_type == 5:
        room_name = "pentagonal"
    elif room_type == 6:
        room_name = "hexagonal"
    elif room_type == 7:
        room_name = "L"
    elif room_type == 8:
        room_name = "T"
    else:
        raise Exception("correct room type: [3~8]")

    lw, h_half, Lnonconvex, Tnonconvex = br.get_size()
    """Lnonconvex=[x,y], Tnonconvex=[x1,x2,y]"""
    corners = br.get_corners(lw, Lnonconvex, Tnonconvex, room_name)
    if conf.build.crushing:
        corners = br.crushing(corners)

    ## LOS check and set src position
    assert conf.build.place_area > 0 and conf.build.place_area <= 1
    if room_name in ["triangle", "shoebox", "pentagonal", "hexagonal"]:
        place_region_corners = conf.build.place_area * corners
        place_region = pra.Room.from_corners(place_region_corners)
    elif room_name == "L":
        pr_margin = np.array([[-0.5], [-0.5]])
        place_region_corners = (conf.build.place_area * corners) + pr_margin
        place_region = pra.Room.from_corners(place_region_corners)
    elif room_name == "T":
        pr_margin = np.array([[0], [0.5]])
        place_region_corners = (conf.build.place_area * corners) + pr_margin
        place_region = pra.Room.from_corners(place_region_corners)
    else:
        raise Exception("Incorret room name")
    ori_region = pra.Room.from_corners(corners)

    pr_xmax = max(place_region_corners[0, :])
    pr_xmin = min(place_region_corners[0, :])
    pr_ymax = max(place_region_corners[1, :])
    pr_ymin = min(place_region_corners[1, :])

    if room_name in ["triangle", "shoebox", "pentagonal", "hexagonal"]:
        src = br.get_src(place_region, ori_region, pr_xmax, pr_xmin, pr_ymax, pr_ymin)
        los = 1  # convex room
    elif room_name == "L":
        src = br.get_src(place_region, ori_region, pr_xmax, pr_xmin, pr_ymax, pr_ymin)
        los = los_checking_L(
            src,
            xmax=corners[0, 3],
            xmin=min(corners[0, :]),
            ymax=corners[1, 3],
            ymin=min(corners[1, :]),
        )
    elif room_name == "T":
        src = br.get_src(place_region, ori_region, pr_xmax, pr_xmin, pr_ymax, pr_ymin)
        los = los_checking_T(
            src,
            xmax=corners[0, 4],
            xmin=corners[0, 1],
            ymax=max(corners[1, :]),
            ymin=max(corners[1, 0:6]),
        )
    else:
        raise Exception("Incorrect room name")

    src_ori = np.expand_dims(src, -1)  # [2,1]
    corners -= src_ori  # src will be located at (0,0,0)

    if conf.build.rotating:
        theta = (np.random.rand(1) * (2 * np.pi)).squeeze()
        corners = br.rotating(corners, theta)

    ## Get labels
    if conf.build.height_fix:
        const_src_h = 1.0
        src_h = np.random.rand(1) * (
            (-h_half + const_src_h) - (-h_half + const_src_h)
        ) + (-h_half + const_src_h)
    else:
        src_h = np.random.rand(1) * (
            (-h_half + conf.build.max_src_h) - (-h_half + conf.build.min_src_h)
        ) + (
            -h_half + conf.build.min_src_h
        )  # absolute position when height=[-h_half, h_half].
    parametric_labels, walls, h_src2floor, h_src2ceiling = br.get_parametric_labels(
        corners, h_half, src_h, room_name
    )  ## labels when src=(0,0,0)

    resolution = conf.build.image_label_resolution
    # print(resolution)
    image_size = 1024  # 1024x1024
    image_labels, filled_image_labels, if_wrong = br.get_image_labels(
        corners, resolution=resolution, size=image_size
    )
    if if_wrong:  # loop breaking
        return if_wrong
    corner_labels = corners.copy()

    assert int(conf.build.max_h * 2 - conf.build.min_src_h) == 4
    height_labels_size = 256
    height_labels = np.ones(height_labels_size * 2)
    filled_floor = int(np.round(abs(h_src2floor) / resolution))
    filled_ceiling = int(np.round(abs(h_src2ceiling) / resolution))
    height_labels[height_labels_size : height_labels_size + filled_floor] = 0
    height_labels[height_labels_size - filled_ceiling : height_labels_size] = 0
    height_labels = height_labels.astype(int)

    if conf.build.FigureOn:
        labelfig(
            filled_image_labels,
            image_size,
            repeat_room,
            resolution=conf.build.image_label_resolution,
        )

    parallel_trans = np.array(
        [[-min(corners[0, :]), -min(corners[1, :])]]
    ).T  # Pyroomacoustics can't simulate RIR when vertex has negarive value
    corners += parallel_trans  # corners = rot(corners_ori - src_ori) + parallel_trans
    src = parallel_trans  # src = src_ori - src_ori + parallel_trans
    src = np.vstack([src, (h_half + src_h)])
    mic_locs = br.get_mic_locs(src_pos=src)

    ## build room
    assert (
        conf.build.use_random_abs + conf.build.use_materials == 1
    )  # select one in [use_randomabs, use_materials]
    if conf.build.use_random_abs:
        assert len(conf.build.abs_coeff) == 2
        abs_coeff = (
            random.random() * (conf.build.abs_coeff[-1] - conf.build.abs_coeff[0])
            + conf.build.abs_coeff[0]
        )
    elif conf.build.use_materials:
        abs_M = pra.Material(
            energy_absorption=random.choice(conf.build.abs_material)
        )  # material for side walls
        abs_M_flo_ceil = pra.make_materials(
            floor=random.choice(conf.build.abs_material_floor),
            ceiling=random.choice(conf.build.abs_material_ceiling),
        )

    if conf.build.add_temp_noise:
        assert len(conf.build.temp_level) == 2
        temperature = (
            random.random() * (conf.build.temp_level[-1] - conf.build.temp_level[0])
            + conf.build.temp_level[0]
        )
    else:
        temperature = 20

    for repeat_order, now_order in enumerate(conf.build.order):
        use_raytracing = conf.build.use_ray_tracing
        if now_order == 1:
            use_raytracing = False

        if not use_raytracing:  # image source model
            if conf.build.use_random_abs:
                room = pra.Room.from_corners(
                    corners,
                    fs=conf.build.fs,
                    max_order=now_order,
                    materials=pra.Material(abs_coeff),
                )
            elif conf.build.use_materials:
                room = pra.Room.from_corners(
                    corners,
                    fs=conf.build.fs,
                    max_order=now_order,
                    materials=abs_M,
                )
        else:  # ray-tracing
            if conf.build.use_random_abs:
                room = pra.Room.from_corners(
                    corners,
                    fs=conf.build.fs,
                    max_order=now_order,
                    materials=pra.Material(abs_coeff, conf.build.scatter_coeff),
                    air_absorption=False,
                    ray_tracing=True,
                )
            elif conf.build.use_materials:
                room = pra.Room.from_corners(
                    corners,
                    fs=conf.build.fs,
                    max_order=now_order,
                    materials=abs_M,
                    air_absorption=False,
                    ray_tracing=True,
                )
            room.set_ray_tracing(
                n_rays=conf.build.nb_rays, receiver_radius=conf.build.receiver_radius
            )

        ## make room to 3D
        if conf.build.use_random_abs:
            room.extrude(
                height=h_half * 2,
                materials=pra.Material(abs_coeff, conf.build.scatter_coeff),
            )
        elif conf.build.use_materials:
            room.extrude(
                height=h_half * 2,
                materials=abs_M_flo_ceil,
            )
        room.add_source(src)
        room.add_microphone_array(mic_locs)

        # ## Fig. room ##
        if conf.build.FigureOn and repeat_order == 0:
            roomfig(room, repeat_room)

        ## compute RIR
        room.compute_rir()
        rt60 = room.measure_rt60().mean()

        if conf.build.remove_direct:
            # if conf.build.use_direct_detection:
            eps_sample = int(0.002 * conf.build.fs)  # 2ms after direct peak

            temp_rir_len = conf.build.fs
            temp_rir = np.zeros([mic_locs.shape[-1], temp_rir_len])
            for i in range(mic_locs.shape[-1]):
                if len(room.rir[i][0]) < temp_rir_len:
                    temp_rir[i, : len(room.rir[i][0])] = room.rir[i][0]
                else:
                    temp_rir[i] = room.rir[i][0][:temp_rir_len]

            ## normalize: direct peak to 1
            temp_rir = temp_rir - np.mean(temp_rir, axis=-1, keepdims=True)
            temp_rir = temp_rir / (
                np.max(np.abs(temp_rir), axis=-1, keepdims=True) + 1e-8
            )

            ## remove direct
            direct_peak_idx = np.argmax(np.abs(temp_rir)[0])
            direct_peak_idx += eps_sample
            rir = temp_rir[:, direct_peak_idx : conf.build.rir_len + direct_peak_idx]

        else:
            raise NotImplementedError("We trimmed direct part of RIR")

        if conf.build.FigureOn and repeat_order == 0:
            rirfig(rir, repeat_room, room_name)

        ## Packing
        rir = torch.Tensor(rir)
        if repeat_order == 0:
            filled_image_labels = torch.Tensor(filled_image_labels)
            height_labels = torch.Tensor(height_labels)
            rt60 = torch.Tensor([rt60])
            # parametric_labels = torch.Tensor(parametric_labels)
            heights = torch.Tensor([h_src2ceiling, h_src2floor])
            # corner_labels = torch.Tensor(corner_labels)
            # walls = torch.Tensor(walls)
            room_type = torch.Tensor(np.expand_dims(room_type, axis=0))
            los = torch.Tensor([los])

        if repeat_order == 0:
            dataset_name = f"{conf.path_dataset}"
        else:
            dataset_name = f"{conf.path_dataset}-{now_order}"
        os.makedirs(dataset_name, exist_ok=True)

        torch.save(
            {
                "rir": rir,
                "filled_image_labels": filled_image_labels,
                "height_labels": height_labels,
                "rt60": rt60,
                # 'parametric_labels': parametric_labels,
                "heights": heights,
                # 'corner_labels': corner_labels,
                # 'walls': walls,
                "room_type": room_type,
                "los": los,
            },
            f"{dataset_name}/{repeat_room}.pt",
        )


def los_checking_L(src, xmax, xmin, ymax, ymin):
    los_check_corners = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    ).T

    los_check = pra.Room.from_corners(los_check_corners)
    los = 1 if los_check.is_inside(src, include_borders=False) else 0
    return los


def los_checking_T(src, xmax, xmin, ymax, ymin):
    los_check_corners = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    ).T

    los_check = pra.Room.from_corners(los_check_corners)
    los = 1 if los_check.is_inside(src, include_borders=False) else 0
    return los


def labelfig(filled_image_labels, image_size, repeat_room, resolution):
    path = "figs/labelfigs/"
    xlabels = [-10, -5, 0, 5, 10]
    ylabels = [10, 5, 0, -5, -10]

    plt.imshow(filled_image_labels, cmap="Greys")
    plt.xticks(
        ticks=[0, image_size * 0.25, image_size * 0.5, image_size * 0.75, image_size],
        labels=xlabels,
    )
    plt.yticks(
        ticks=[0, image_size * 0.25, image_size * 0.5, image_size * 0.75, image_size],
        labels=ylabels,
    )
    plt.xlabel("pixel: {} m".format(resolution))

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "{}".format(repeat_room))
    plt.close()


def roomfig(room, repeat_room):
    path = "figs/roomfigs/"
    fig, ax = room.plot()
    ax.set_xlim([-2, 12])
    ax.set_xlabel("X")
    ax.set_ylim([-2, 12])
    ax.set_ylabel("Y")
    ax.set_zlim([-1, 6])
    ax.set_zlabel("Z")

    ax.view_init(elev=90, azim=270)
    # ax.view_init(elev=0, azim=270)
    # ax.view_init(elev=30, azim=30)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "{}".format(repeat_room), bbox_inches="tight")
    plt.close()


def rirfig(rir, repeat_room, room_name, plot_view="1D"):
    path = "figs/rirfigs/"
    if plot_view == "1D":
        for ch in range(rir.shape[0]):
            plt.plot(rir[ch, :])
    elif plot_view == "2D":
        shw = plt.imshow(abs(rir[:, :]), aspect="auto")
        plt.colorbar(shw)
    else:
        raise Exception("select plot_view in [1D, 2D]")
    plt.title("roomtype: {}".format(room_name))

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "{}".format(repeat_room), dpi=300)
    plt.close()


if __name__ == "__main__":
    conf = {
        "path_dataset": "dataset/train-456lt",
        "build": {
            "FigureOn": False,
            "nb_dataset": [0, 200000],
            "room_types": [4, 5, 6, 7, 8],  # [quad., penta., hexa., L, T]
            "room_prob": [1 / 7, 1 / 7, 1 / 7, 2 / 7, 2 / 7],
            "classes": 8 + 2,  # max(side-walls) + floor + ceiling
            "min_lw": 2,
            "max_lw": 5,
            "min_h": 1.5,
            "max_h": 2.5,
            "radius": 0.05,
            "height_fix": False,
            "min_src_h": 1,  # 1m from floor
            "max_src_h": 1.5,
            "fs": 8000,
            "rir_len": 1024,
            "order": [6],
            "mic_type": "ring",
            "label_permutation": False,
            "label_positive": False,
            "image_label_resolution": 0.02,  # 2cm
            "crushing": True,
            "crushing_level": 0.25,
            "rotating": True,
            "use_random_abs": False,
            "abs_coeff": [0.1, 0.3],
            "use_materials": True,
            "abs_material": [
                "hard_surface",
                "glass_window",
                "rough_concrete",
                "rough_lime_wash",
                "plasterboard",
            ],
            "abs_material_floor": [
                "linoleum_on_concrete",
                "carpet_thin",
                "audience_floor",
            ],
            "abs_material_ceiling": [
                "ceiling_perforated_gypsum_board",
                "ceiling_metal_panel",
                "ceiling_plasterboard",
            ],
            "scatter_coeff": 0.1,
            "use_ray_tracing": True,
            "nb_rays": 10000,
            "receiver_radius": 0.5,
            "place_area": 0.7,
            "remove_direct": True,
            "add_temp_noise": False,
            "temp_level": [15, 25],
        },
    }
    conf = ConfigWrapper(**conf)
    assert len(conf.build.room_types) == len(conf.build.room_prob)

    if conf.build.mic_type == "eigenmike":
        # mic_angle = np.load('eigenmike_pos_phi_theta.npy')
        # r = 0.042 # same as Eigenmike
        raise Exception("we not consider eigenmike type now")
    elif conf.build.mic_type == "ring":
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
        r = conf.build.radius
    else:
        raise Exception("choose correct mic type; eigenmike, ring")

    br = BuildRoom(conf, r, mic_angle)

    nb_dataset_start = conf.build.nb_dataset[0]
    nb_dataset_end = conf.build.nb_dataset[1]
    with tqdm(total=nb_dataset_end - nb_dataset_start) as pbar:
        while nb_dataset_start < nb_dataset_end:
            if_wrong = main(nb_dataset_start)
            if if_wrong:  # True means out-of-bound
                continue
            nb_dataset_start += 1
            pbar.update(1)
