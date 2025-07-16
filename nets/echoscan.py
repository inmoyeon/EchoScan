import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()

        self.Encoder = Encoder(conf)
        self.Decoder = Decoder(conf)
        self.HeightExtractor = HeightExtractor(conf)

    def forward(self, X):
        encoded_X, p_vals = self.Encoder(X)  # [B, D]
        image = self.Decoder(encoded_X)
        height = self.HeightExtractor(encoded_X)

        outs = {"image": image, "height": height, "p_vals": p_vals}

        return outs


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, "the input tensor of L2Norm must be the shape of [B, C]"
        return F.normalize(x, p=2, dim=-1, eps=1e-8)


class Encoder(nn.Module):
    def __init__(self, conf):
        super(Encoder, self).__init__()
        self.conf = conf

        input_ch = conf.model.input_ch
        init_ch = conf.model.init_ch

        init_kernel = int(conf.model.fs * 0.001)  # 1ms kernel
        if init_kernel % 2 == 0:
            init_kernel += 1

        self.preconv = nn.Conv1d(
            input_ch,
            init_ch,
            kernel_size=init_kernel,
            stride=2,
            padding=init_kernel // 2,
        )  # for rir 8k and length 1024
        self.preconv_norm = Normalize1d(init_ch)

        self.cb1 = nn.Sequential(
            ConvBlock(conf, init_ch, ch_expand=False),
            ConvBlock(conf, init_ch, ch_expand=False),
            ConvBlock(conf, init_ch, ch_expand=False),
        )
        self.cb2 = nn.Sequential(
            ConvBlock(conf, init_ch, ch_expand=True),
            ConvBlock(conf, init_ch * 2, ch_expand=False),
        )
        self.cb3 = nn.Sequential(
            ConvBlock(conf, init_ch * 2, ch_expand=True),
            ConvBlock(conf, init_ch * 4, ch_expand=False),
        )
        self.cb4 = nn.Sequential(
            ConvBlock(conf, init_ch * 4, ch_expand=True),
            ConvBlock(conf, init_ch * 8, ch_expand=False),
            ConvBlock(conf, init_ch * 8, ch_expand=False),
        )
        self.cb5 = nn.Sequential(
            ConvBlock(conf, init_ch * 8, ch_expand=True),
            ConvBlock(conf, init_ch * 16, ch_expand=False),
            ConvBlock(conf, init_ch * 16, ch_expand=False),
        )
        self.cb6 = nn.Sequential(
            ConvBlock(conf, init_ch * 16, ch_expand=True),
            ConvBlock(conf, init_ch * 32, ch_expand=False),
            ConvBlock(conf, init_ch * 32, ch_expand=False),
        )

        if conf.model.use_trainable_gdescriptor:
            self.p0 = nn.Parameter(torch.zeros([]))
            self.p1 = nn.Parameter(torch.ones([]))
        else:
            self.p0 = 1
            self.p1 = 3

        self.linear_g0 = nn.Sequential(nn.Linear(1024, 256, bias=False), L2Norm())
        self.linear_g1 = nn.Sequential(nn.Linear(1024, 256, bias=False), L2Norm())

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_norm(x)
        x = nonlinearity(x)

        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.cb5(x)
        x = self.cb6(x)

        if self.conf.model.use_trainable_gdescriptor:
            gd0 = GlobalDescriptor(p=1 + F.relu(self.p0))(x)
            gd1 = GlobalDescriptor(p=1 + F.relu(self.p1))(x)
        else:
            gd0 = GlobalDescriptor(p=self.p0)(x)
            gd1 = GlobalDescriptor(p=self.p1)(x)

        assert (
            gd0.dim() == 2 and gd1.dim() == 2
        ), "the output tensor of GlobalDescriptor must be the shape of [B, C]"
        gd0 = self.linear_g0(gd0)
        gd1 = self.linear_g1(gd1)

        gd = torch.cat([gd0, gd1], dim=1)
        gd = F.normalize(gd, dim=-1, eps=1e-8)
        return gd, [self.p0, self.p1]


class ConvBlock(nn.Module):
    def __init__(self, conf, in_ch, ch_expand):
        super(ConvBlock, self).__init__()

        self.conf = conf
        self.ch_expand = ch_expand
        kernel, pad = 5, 2
        if self.ch_expand:
            self.conv1 = nn.Conv1d(in_ch, in_ch * 2, kernel, stride=2, padding=pad)
            self.norm1 = Normalize1d(in_ch * 2)
            self.conv2 = nn.Conv1d(in_ch * 2, in_ch * 2, kernel, stride=1, padding=pad)
            self.norm2 = Normalize1d(in_ch * 2)
            self.proj_block = nn.Conv1d(in_ch, in_ch * 2, 1, stride=2)
        else:
            self.conv1 = nn.Conv1d(in_ch, in_ch, kernel, stride=1, padding=pad)
            self.norm1 = Normalize1d(in_ch)
            self.conv2 = nn.Conv1d(in_ch, in_ch, kernel, stride=1, padding=pad)
            self.norm2 = Normalize1d(in_ch)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nonlinearity(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = nonlinearity(x)

        if self.ch_expand:
            identity = self.proj_block(identity)

        x += identity
        return x


class Decoder(torch.nn.Module):
    def __init__(self, conf):
        super(Decoder, self).__init__()
        self.conf = conf
        self.reshape_lin = nn.Sequential(
            nn.Linear(in_features=256, out_features=64 * 16 * 16),
            nn.ReLU6(),
        )

        self.dcb1 = SimpleDecoder(128, 64, upsample_level=2)
        self.dcb2 = SimpleDecoder(64, 64, upsample_level=2)
        self.res_dcb2 = SimpleDecoder(128, 1, upsample_level=4)

        self.dcb3 = SimpleDecoder(64, 64, upsample_level=2)
        self.dcb4 = SimpleDecoder(64, 64, upsample_level=2)
        self.res_dcb4 = SimpleDecoder(128, 1, upsample_level=16)

        self.dcb5 = SimpleDecoder(64, 32, upsample_level=2)
        self.dcb6 = SimpleDecoder(32, 32, upsample_level=2)
        self.dcb_out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = torch.chunk(x, 2, dim=1)
        x = torch.stack(x, dim=1)

        x = self.reshape_lin(x)
        x = x.reshape(-1, 64 * 2, 16, 16).contiguous()
        identity = x

        x = self.dcb1(x)
        x = self.dcb2(x)
        res_dcb2 = self.res_dcb2(identity)
        x += res_dcb2

        x = self.dcb3(x)
        x = self.dcb4(x)
        res_dcb4 = self.res_dcb4(identity)
        x += res_dcb4

        x = self.dcb5(x)
        x = self.dcb6(x)
        x = self.dcb_out(x)
        x = x.squeeze(1)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_level=2):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=upsample_level, mode="nearest"),
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout=0.0,
            ),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        x = self.dec(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = nonlinearity(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = nonlinearity(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True, upsample_level=2):
        super().__init__()
        self.with_conv = with_conv
        self.upsample_level = upsample_level
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        self.upsample = nn.Upsample(scale_factor=upsample_level, mode="nearest")

    def forward(self, x):
        x = self.upsample(x)

        if self.with_conv:
            x = self.conv(x)
        return x


class HeightExtractor(nn.Module):  # in: Bx1024
    def __init__(self, conf):
        super(HeightExtractor, self).__init__()
        self.conf = conf

        gd_channel = 512
        out_channel = 512

        self.hdcb1 = nn.Sequential(nn.Linear(gd_channel, out_channel), nn.Sigmoid())

    def forward(self, x):
        x = self.hdcb1(x)
        return x


## Misc. ##
class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.eps = 1e-8

    def forward(self, x):
        assert (
            x.dim() == 3
        ), "the input tensor of GlobalDescriptor must be the shape of [B, C, D]"
        if self.p == 1:
            return x.mean(dim=[-1])
        elif self.p == float("inf"):
            return torch.flatten(
                F.adaptive_max_pool2d(x, output_size=(1024, 1)), start_dim=2
            )
        else:
            sum_value = x.pow(self.p).mean(dim=[-1])
            return torch.sign(sum_value) * (
                (torch.abs(sum_value + self.eps) + self.eps).pow(1.0 / self.p)
            )

    def extra_repr(self):
        return "p={}".format(self.p)


def nonlinearity(x):
    return F.relu6(x)


def Normalize(in_channels):
    return nn.BatchNorm2d(in_channels)


def Normalize1d(in_channels):
    return nn.BatchNorm1d(in_channels)


def LayerNormalize(x, shapes):
    return F.layer_norm(x, shapes[1:])
