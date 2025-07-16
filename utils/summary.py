from torch.utils.tensorboard import SummaryWriter

## tensorboard --logdir=runs --port=xxxx


def draw_tensorboard_value(conf, tag, value, epoch):
    path = conf.path_logdir
    writer = SummaryWriter(path)
    writer.add_scalar(tag, value, epoch)


def draw_tensorboard(conf, losses, set_name, epoch, etc=None):
    """
    losses = dictionary format
    etc = [lr, ...]

    if "AttributeError: module 'distutils' has no attribute 'version'" is occurred,
    use this commend below:
    pip install setuptools==59.5.0
    """
    path = conf.path_logdir
    writer = SummaryWriter(path)

    ## Main losses ##
    # writer.add_scalar('Train/Learning_rate'.format(set_name), etc[0], epoch)
    writer.add_scalar("{}/0-Total".format(set_name), losses["loss_total"], epoch)
    writer.add_scalar("{}/1-Image".format(set_name), losses["loss_image"], epoch)
    writer.add_scalar(
        "{}/2-Image_dice".format(set_name), losses["loss_image_dice"], epoch
    )
    writer.add_scalar("{}/3-Height".format(set_name), losses["loss_height"], epoch)

    ## Split-type losses ##
    writer.add_scalar(
        "{}-split-total/0_shoebox".format(set_name),
        losses["loss_total_split"][0].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/1_penta.".format(set_name),
        losses["loss_total_split"][1].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/2_hexa.".format(set_name),
        losses["loss_total_split"][2].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/3_L(LOS)".format(set_name),
        losses["loss_total_split"][3].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/4_L(NLOS)".format(set_name),
        losses["loss_total_split"][4].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/5_T(LOS)".format(set_name),
        losses["loss_total_split"][5].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-total/6_T(NLOS)".format(set_name),
        losses["loss_total_split"][6].item(),
        epoch,
    )

    writer.add_scalar(
        "{}-split-image/0_shoebox".format(set_name),
        losses["loss_image_split"][0].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/1_penta.".format(set_name),
        losses["loss_image_split"][1].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/2_hexa.".format(set_name),
        losses["loss_image_split"][2].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/3_L(LOS)".format(set_name),
        losses["loss_image_split"][3].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/4_L(NLOS)".format(set_name),
        losses["loss_image_split"][4].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/5_T(LOS)".format(set_name),
        losses["loss_image_split"][5].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image/6_T(NLOS)".format(set_name),
        losses["loss_image_split"][6].item(),
        epoch,
    )

    writer.add_scalar(
        "{}-split-image_dice/0_shoebox".format(set_name),
        losses["loss_image_dice_split"][0].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/1_penta.".format(set_name),
        losses["loss_image_dice_split"][1].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/2_hexa.".format(set_name),
        losses["loss_image_dice_split"][2].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/3_L(LOS)".format(set_name),
        losses["loss_image_dice_split"][3].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/4_L(NLOS)".format(set_name),
        losses["loss_image_dice_split"][4].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/5_T(LOS)".format(set_name),
        losses["loss_image_dice_split"][5].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-image_dice/6_T(NLOS)".format(set_name),
        losses["loss_image_dice_split"][6].item(),
        epoch,
    )

    writer.add_scalar(
        "{}-split-height/0_shoebox".format(set_name),
        losses["loss_height_split"][0].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/1_penta.".format(set_name),
        losses["loss_height_split"][1].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/2_hexa.".format(set_name),
        losses["loss_height_split"][2].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/3_L(LOS)".format(set_name),
        losses["loss_height_split"][3].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/4_L(NLOS)".format(set_name),
        losses["loss_height_split"][4].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/5_T(LOS)".format(set_name),
        losses["loss_height_split"][5].item(),
        epoch,
    )
    writer.add_scalar(
        "{}-split-height/6_T(NLOS)".format(set_name),
        losses["loss_height_split"][6].item(),
        epoch,
    )
