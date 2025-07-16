from configwrapper import ConfigWrapper
from nets.net import RoomGeometryInference
from data_loader import EchoscanDataset
from utils.summary import draw_tensorboard

import os
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

## Set PrintOptions ##
set_printoptions = True
if set_printoptions:
    np.set_printoptions(precision=8, suppress=True, threshold=np.inf)
    torch.set_printoptions(precision=8, sci_mode=False, threshold=np.inf)


def train(model, loader_train, optimizer, epoch):
    global train_losses

    model.train()
    for batch_idx, batch in enumerate(tqdm(loader_train)):

        X = batch["rir"].to(device)
        Y_image = batch["image_label"].to(device)  # [B, 1024, 1024]
        Y_height_v = batch["height_label"].to(device)  # [B, 512]
        Y_roomtype = batch["room_type"].to(device)
        Y_los = batch["los"].to(device)

        optimizer.zero_grad()
        if conf.use_amp:
            with autocast():
                outs, losses = model(X, Y_image, Y_height_v, Y_roomtype, Y_los)
                losses = arranging_losses(losses, nb_device)
                loss = losses["loss_total"]
        else:
            outs, losses = model(X, Y_image, Y_height_v, Y_roomtype, Y_los)
            losses = arranging_losses(losses, nb_device)
            loss = losses["loss_total"]

        if conf.use_amp:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if conf.train.use_scheduler:
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            lr = conf.train.init_lr

        ## print during training ##
        if batch_idx % 20 == 0:
            print(
                f"\n\n<Batch: {batch_idx}>\
                \n lr: {lr}\
                \n loss_tot: {losses['loss_total']}\
                \n loss_image: {losses['loss_image']}\
                \n loss_image_dice: {losses['loss_image_dice']}\
                \n loss_height: {losses['loss_height']}"
            )

        ## batch-step-losses ##
        assert len(losses.keys()) == len(train_losses.keys())
        for key in losses.keys():
            if "_split" in key:
                train_losses[key] += losses[key].detach().cpu()
            else:
                train_losses[key] += losses[key].detach().cpu().item()

    ## epoch-step-losses ##
    for key in losses.keys():
        train_losses[key] /= batch_idx + 1

    draw_tensorboard(
        conf,
        losses=train_losses,
        set_name="Train",
        epoch=epoch,
        etc=[lr],
    )


def valid(model, loader_valid, optimizer, epoch):
    global valid_losses
    global best_epoch, best_losses

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader_valid)):

            X = batch["rir"].to(device)
            Y_image = batch["image_label"].to(device)  # [B, 1024, 1024]
            Y_height_v = batch["height_label"].to(device)  # [B, 512]
            Y_roomtype = batch["room_type"].to(device)
            Y_los = batch["los"].to(device)

            if conf.use_amp:
                with autocast():
                    outs, losses = model(X, Y_image, Y_height_v, Y_roomtype, Y_los)
                    losses = arranging_losses(losses, nb_device)
                    out_image, out_height = outs["image"], outs["height"]
            else:
                outs, losses = model(X, Y_image, Y_height_v, Y_roomtype, Y_los)
                losses = arranging_losses(losses, nb_device)
                out_image, out_height = outs["image"], outs["height"]

            ## batch-step-losses ##
            assert len(losses.keys()) == len(valid_losses.keys())
            for key in losses.keys():
                if "_split" in key:
                    valid_losses[key] += losses[key].detach().cpu()
                else:
                    valid_losses[key] += losses[key].detach().cpu().item()

    ## epoch-step-losses ##
    for key in losses.keys():
        valid_losses[key] /= batch_idx + 1
    print(
        f"\nvalid_loss: {valid_losses['loss_total']:.4f}\
        / {valid_losses['loss_image']:.4f}\
        / {valid_losses['loss_image_dice']:.4f}\
        / {valid_losses['loss_height']:.4f}\
        \n{valid_losses['loss_total_split']}\
        \n{valid_losses['loss_image_split']}\
        \n{valid_losses['loss_image_dice_split']}\
        \n{valid_losses['loss_height_split']}"
    )

    draw_tensorboard(
        conf,
        losses=valid_losses,
        set_name="Valid",
        epoch=epoch,
    )

    ## Best ##
    if valid_losses["loss_total"] < best_losses["loss_total"]:
        best_epoch = epoch
        best_losses = valid_losses

        if nb_device > 1:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.module.state_dict(),
                },
                path_logdir + "checkpoint{}.pt".format(best_epoch),
            )
        else:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                },
                path_logdir + "checkpoint{}.pt".format(best_epoch),
            )
        print(
            f"\n\nbest_loss: {best_losses['loss_total']:.4f}\
            / {best_losses['loss_image']:.4f}\
            / {best_losses['loss_image_dice']:.4f}\
            / {best_losses['loss_height']:.4f}\
            \n{best_losses['loss_total_split']}\
            \n{best_losses['loss_image_split']}\
            \n{best_losses['loss_image_dice_split']}\
            \n{best_losses['loss_height_split']}"
        )

        draw_tensorboard(
            conf,
            losses=best_losses,
            set_name="Best",
            epoch=best_epoch,
        )

    elif epoch % 20 == 0:
        if nb_device > 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),  #
                },
                path_logdir + "checkpoint{}_noval.pt".format(epoch),
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                path_logdir + "checkpoint{}_noval.pt".format(epoch),
            )


def arranging_losses(losses, nb_device):
    """final loss shape -> loss_*: [], loss_*_split: [7]"""
    if nb_device > 1:
        for key in losses.keys():
            if "_split" in key:
                losses[key] = losses[key].mean(dim=0)
            else:
                losses[key] = losses[key].mean()
    else:
        for key in losses.keys():
            if "_split" in key:
                losses[key] = losses[key].squeeze()

    return losses


if __name__ == "__main__":
    conf = {
        "gpu": "1",
        # "training_type": "new",  # [new, continue, finetune]
        "training_type": "continue",  # [new, continue, finetune]
        "path_ckpt": "runs/echoscan_gradnorm/checkpoint6.pt",
        "path_dataset_train": "dataset/train-456lt",
        "path_dataset_valid": "dataset/val-456lt",
        "path_logdir": "runs/echoscan_gradnorm/",
        "use_amp": True,
        "loader": {
            "add_g_noise": True,
            "tr_snr_range": [0, 20],
            "val_snr_range": [10, 20],
            "num_workers": 4,
            "use_pin_memory": True,
            "use_persistent_workers": True,
            "use_normalize": False,
            "normalize_type": "zscore",
            "augmentation": [
                "time_masking",
                # "none",
            ],
        },
        "train": {
            "epoch": 300,
            "batch_size": 32,
            "use_scheduler": True,
            "init_lr": 1e-3,
            # "min_lr": 1e-5,
            "min_lr": 1e-4,
        },
        "model": {
            "module": "echoscan",
            "image_loss_type": "mse",
            "height_loss_type": "mse",
            # "loss_w_dice": 0.3,
            "loss_w_dice": 0.5,
            "loss_w_height": 1.0,
            "use_pit": True,
            "use_trainable_gdescriptor": False,
            "fs": 8000,
            "init_ch": 32,
            "input_ch": 6,
            "resolution": 0.02,  # inter-pixel distance
        },
    }
    conf = ConfigWrapper(**conf)

    ## GPU ##
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_device = torch.cuda.device_count()

    ## Params ###
    path_logdir = conf.path_logdir
    os.makedirs(path_logdir, exist_ok=True)

    print("which device: ", conf.gpu)
    print("starting lr: ", conf.train.init_lr)
    print("data path: ", conf.path_dataset_train)
    print("params path: ", path_logdir)

    ## Loader ##
    ds_train = EchoscanDataset(
        conf, split="train", augmentation=conf.loader.augmentation
    )
    ds_valid = EchoscanDataset(conf, split="valid")

    batch_size = (
        conf.train.batch_size * nb_device if nb_device != 0 else conf.train.batch_size
    )
    workers = (
        conf.loader.num_workers * nb_device
        if nb_device != 0
        else conf.loader.num_workers
    )
    assert batch_size > 1
    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=conf.loader.use_persistent_workers,
        pin_memory=conf.loader.use_pin_memory,
    )
    loader_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        persistent_workers=conf.loader.use_persistent_workers,
        pin_memory=conf.loader.use_pin_memory,
    )

    ## Make Model ##
    model = RoomGeometryInference(conf)
    if nb_device > 1:
        model = nn.DataParallel(model)

    if conf.training_type == "new":
        start_epoch = 0
    elif conf.training_type == "continue":
        ckpt = torch.load(conf.path_ckpt, map_location="cpu")
        start_epoch = ckpt["epoch"]
        start_epoch += 1
        ckpt = ckpt["model_state_dict"]
    elif conf.training_type == "finetune":
        ckpt = torch.load(conf.path_ckpt, map_location="cpu")
        start_epoch = 0
        ckpt = ckpt["model_state_dict"]
    else:
        raise ValueError("training_type is not valid [new, continue, finetune]")

    ## ckpt filtering
    if nb_device > 1 and not conf.training_type == "new":
        module_ckpt = {}
        for key in ckpt.keys():
            module_ckpt[f"module.{key}"] = ckpt[key]
        ckpt = module_ckpt

    if not conf.training_type == "new":
        keys_not_in_model = []
        keys_size_mismatch = []
        current_state_dict = model.state_dict()
        print(
            f"State dict key size:\n\
            Current: {len(list(current_state_dict.keys()))}\n\
            Checkpoint: {len(list(ckpt.keys()))}"
        )
        for key in list(ckpt.keys()):
            if key not in current_state_dict.keys():
                keys_not_in_model.append(key)
                del ckpt[key]
                continue

            if current_state_dict[key].size() != ckpt[key].size():
                del ckpt[key]
                keys_size_mismatch.append(key)
        print(
            f"\n\
            Params Not in the model: {keys_not_in_model}\n\
            Params Size Mismatch: {keys_size_mismatch}\
        "
        )
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf.train.init_lr)
    if conf.train.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=(ds_train.__len__() // batch_size),
            T_mult=1,
            eta_min=conf.train.min_lr,
        )

    if conf.use_amp:
        scaler = GradScaler()

    ## Training ##
    best_epoch = start_epoch
    best_losses = {"loss_total": 100}
    for epoch in tqdm(range(start_epoch, start_epoch + conf.train.epoch)):
        train_losses = {
            "loss_total": 0,
            "loss_image": 0,
            "loss_image_dice": 0,
            "loss_height": 0,
            "loss_total_split": torch.zeros(7),
            "loss_image_split": torch.zeros(7),
            "loss_image_dice_split": torch.zeros(7),
            "loss_height_split": torch.zeros(7),
        }
        valid_losses = copy.deepcopy(train_losses)

        train(model, loader_train, optimizer, epoch)
        valid(model, loader_valid, optimizer, epoch)

    print(
        f"\n\n<< BEST >>\
        \nepoch: {best_epoch}\
        \nBest_loss: {best_losses['loss_total']:.4f}\
        / {best_losses['loss_image']:.4f}\
        / {best_losses['loss_image_dice']:.4f}\
        / {best_losses['loss_height']:.4f}\
        \n{best_losses['loss_total_split']}\
        \n{best_losses['loss_image_split']}\
        \n{best_losses['loss_image_dice_split']}\
        \n{best_losses['loss_height_split']}"
    )
