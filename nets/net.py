import nets.echoscan as echoscan
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = 1 - inputs
        targets = 1 - targets

        intersection = (inputs * targets).sum(dim=-1)
        dice = (2.0 * intersection + smooth) / (
            inputs.sum(dim=-1) + targets.sum(dim=-1) + smooth
        )

        return 1 - dice


class RoomGeometryInference(nn.Module):
    def __init__(self, conf):
        super(RoomGeometryInference, self).__init__()
        self.conf = conf

        ## Model
        self.Model = echoscan.Model(conf)

        ## Loss
        if conf.model.image_loss_type == "mse":
            self.ImageLoss = nn.MSELoss(reduction="none")
        elif conf.model.image_loss_type == "bce":
            self.ImageLoss = nn.BCELoss(reduction="none")

        if conf.model.height_loss_type == "mse":
            self.HeightLoss = nn.MSELoss(reduction="none")
        elif conf.model.height_loss_type == "bce":
            self.HeightLoss = nn.BCELoss(reduction="none")
        self.diceloss = DiceLoss()

    def forward(self, X, Y_image, Y_height, Y_roomtype, Y_los):
        outs = self.Model(X)
        losses = self.compute_loss(outs["image"], Y_image, outs["height"], Y_height)
        losses = self.split_loss(losses, roomtype=Y_roomtype, los=Y_los)

        for key in losses.keys():  # reducing batch size, except for split-type losses
            if not "_split" in key:
                losses[key] = losses[key].mean()

        return outs, losses

    def compute_loss(self, out_image, Y_image, out_height, Y_height):
        """
        calculate various losses without reducing batch size
        note that, keep the shape with batch size
        """
        B = Y_image.shape[0]
        Y_image = Y_image.reshape(B, -1).contiguous()
        out_image = out_image.reshape(B, -1).contiguous()
        Y_height = Y_height.reshape(B, -1).contiguous()
        out_height = out_height.reshape(B, -1).contiguous()

        loss_image = self.ImageLoss(out_image, Y_image).mean(dim=-1)  # [B]
        loss_image_dice = self.diceloss(out_image, Y_image)  # [B]

        ## PIT for height : ring array ##
        if self.conf.model.use_pit:
            out_height_flipped = torch.flip(out_height, [1])  # [B,H]
            loss_height_unflipped = self.HeightLoss(out_height, Y_height).mean(
                dim=-1
            )  # [B]
            loss_height_flipped = self.HeightLoss(out_height_flipped, Y_height).mean(
                dim=-1
            )  # [B]

            out_height_stack = torch.stack(
                [out_height, out_height_flipped], dim=1
            )  # [B,2, H]
            loss_height_stack = torch.stack(
                [loss_height_unflipped, loss_height_flipped], dim=1
            )  # [B,2]
            min_idx = torch.argmin(loss_height_stack, dim=1, keepdim=True)  # [B,1]

            loss_height = torch.gather(loss_height_stack, dim=1, index=min_idx).squeeze(
                dim=1
            )  # [B]
            out_height = out_height.new_zeros(out_height.shape)  # [B,H]
            for b in range(B):
                out_height[b] = out_height_stack[b, min_idx[b, 0]]

        else:
            loss_height = self.HeightLoss(out_height, Y_height).mean(dim=-1)

        loss_total = (
            loss_image
            + (self.conf.model.loss_w_dice * loss_image_dice)
            + (self.conf.model.loss_w_height * loss_height)
        )

        losses = {
            "loss_total": loss_total,
            "loss_image": loss_image,
            "loss_image_dice": loss_image_dice,
            "loss_height": loss_height,
        }

        return losses

    def split_loss(self, losses, roomtype, los):
        """
        unsqueeze(dim=0) : make shape as [1,?] -> considering multi-gpu system
        nan_to_num() : NaN is occurred when there is no roomtype in mini-batch
        """
        sel14 = (roomtype == 4).squeeze(-1)
        sel15 = (roomtype == 5).squeeze(-1)
        sel16 = (roomtype == 6).squeeze(-1)
        sel17 = torch.logical_and(los == 1, roomtype == 7).squeeze(-1)
        sel07 = torch.logical_and(los == 0, roomtype == 7).squeeze(-1)
        sel18 = torch.logical_and(los == 1, roomtype == 8).squeeze(-1)
        sel08 = torch.logical_and(los == 0, roomtype == 8).squeeze(-1)

        loss_total_split = (
            losses["loss_total"]
            .new_tensor(
                [
                    losses["loss_total"][sel14].mean(),
                    losses["loss_total"][sel15].mean(),
                    losses["loss_total"][sel16].mean(),
                    losses["loss_total"][sel17].mean(),
                    losses["loss_total"][sel07].mean(),
                    losses["loss_total"][sel18].mean(),
                    losses["loss_total"][sel08].mean(),
                ]
            )
            .unsqueeze(dim=0)
            .nan_to_num()
        )
        losses["loss_total_split"] = loss_total_split

        loss_image_split = (
            losses["loss_image"]
            .new_tensor(
                [
                    losses["loss_image"][sel14].mean(),
                    losses["loss_image"][sel15].mean(),
                    losses["loss_image"][sel16].mean(),
                    losses["loss_image"][sel17].mean(),
                    losses["loss_image"][sel07].mean(),
                    losses["loss_image"][sel18].mean(),
                    losses["loss_image"][sel08].mean(),
                ]
            )
            .unsqueeze(dim=0)
            .nan_to_num()
        )
        losses["loss_image_split"] = loss_image_split

        loss_image_dice_split = (
            losses["loss_image_dice"]
            .new_tensor(
                [
                    losses["loss_image_dice"][sel14].mean(),
                    losses["loss_image_dice"][sel15].mean(),
                    losses["loss_image_dice"][sel16].mean(),
                    losses["loss_image_dice"][sel17].mean(),
                    losses["loss_image_dice"][sel07].mean(),
                    losses["loss_image_dice"][sel18].mean(),
                    losses["loss_image_dice"][sel08].mean(),
                ]
            )
            .unsqueeze(dim=0)
            .nan_to_num()
        )
        losses["loss_image_dice_split"] = loss_image_dice_split

        loss_height_split = (
            losses["loss_height"]
            .new_tensor(
                [
                    losses["loss_height"][sel14].mean(),
                    losses["loss_height"][sel15].mean(),
                    losses["loss_height"][sel16].mean(),
                    losses["loss_height"][sel17].mean(),
                    losses["loss_height"][sel07].mean(),
                    losses["loss_height"][sel18].mean(),
                    losses["loss_height"][sel08].mean(),
                ]
            )
            .unsqueeze(dim=0)
            .nan_to_num()
        )
        losses["loss_height_split"] = loss_height_split

        return losses
