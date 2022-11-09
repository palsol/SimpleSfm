import torch
from torch import nn
from typing import Optional


class LayerNormTotal(nn.Module):
    """LayerNorm that acts on all the axes except the batch axis"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNormTotal, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.rand(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.shape[0] == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.shape[0], -1).mean(1).view(*shape)
            std = x.view(x.shape[0], -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class LayerNormChannels(nn.LayerNorm):
    """LayerNorm that acts on the channels dimension only (i.e. for tensors of shape B x C x ...)"""

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        permutation = [0] + list(range(2, x.dim())) + [1]  # e.g., for images: [0, 2, 3, 1]
        x = super().forward(x.permute(*permutation).contiguous())
        permutation = [0, x.dim() - 1] + list(range(1, x.dim() - 1))  # e.g., for images: [0, 3, 1, 2]
        return x.permute(*permutation).contiguous()


UPSAMPLING_TYPES = {'nearest': 1, 'linear': 1, 'bilinear': 2, 'bicubic': 2, 'trilinear': 3}


def get_norm_layer(norm: str,
                   channels: int,
                   dim: int = 2,
                   ) -> Optional[nn.Module]:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    module = None

    if norm == 'bn':
        if dim == 1:
            module = nn.BatchNorm1d(channels)
        elif dim == 2:
            module = nn.BatchNorm2d(channels)
        elif dim == 3:
            module = nn.BatchNorm3d(channels)

    elif norm == 'in':
        if dim == 1:
            module = nn.InstanceNorm1d(channels)
        elif dim == 2:
            module = nn.InstanceNorm2d(channels)
        elif dim == 3:
            module = nn.InstanceNorm3d(channels)

    elif norm == 'ln':
        module = LayerNormTotal(channels)

    elif norm == 'ln-channels':
        module = LayerNormChannels(channels)

    elif norm in {'none'}:
        module = None

    else:
        raise ValueError(f'Unsupported normalization: {norm}')

    return module


def get_padding_layer(pad_type: str,
                      padding: int,
                      dim: int = 2,
                      ) -> nn.Module:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    module = None

    if pad_type == 'reflect':
        if dim == 1:
            module = nn.ReflectionPad1d(padding)
        elif dim == 2:
            module = nn.ReflectionPad2d(padding)
        elif dim == 3:
            raise ValueError(f'Pytorch does not have ReflectionPad3d module')

    elif pad_type == 'replicate':
        if dim == 1:
            module = nn.ReplicationPad1d(padding)
        elif dim == 2:
            module = nn.ReplicationPad2d(padding)
        elif dim == 3:
            module = nn.ReplicationPad3d(padding)

    elif pad_type == 'zero':
        if dim == 1:
            module = nn.ConstantPad1d(padding, 0.)
        elif dim == 2:
            module = nn.ConstantPad2d(padding, 0.)
        elif dim == 3:
            module = nn.ConstantPad3d(padding, 0.)

    else:
        raise ValueError(f'Unsupported padding type {pad_type}')

    return module


def get_upsample_layer(scale_factor: int,
                       dim: int,
                       mode: str = None,
                       align_corners: bool = False) -> nn.Module:
    if dim not in {1, 2, 3}:
        raise ValueError(f'Unsupported dim={dim}')

    if mode is None:
        if dim == 1:
            mode = 'linear'
        elif dim == 2:
            mode = 'bilinear'
        elif dim == 3:
            mode = 'trilinear'

    if mode not in UPSAMPLING_TYPES.keys():
        raise ValueError(f'Upsampling type: {mode} is unsupported \
               but should be one of the following: {UPSAMPLING_TYPES}')
    if UPSAMPLING_TYPES[mode] != dim:
        raise ValueError(f'Upsampling type: {mode} is wrong \
                           for dim: {dim} ')

    module = nn.Upsample(scale_factor=scale_factor,
                         mode=mode,
                         align_corners=align_corners)
    return module


def set_weight_normalization(module: nn.Module,
                             norm: Optional[str]) -> nn.Module:
    if norm == 'sn':
        return nn.utils.spectral_norm(module)

    elif norm in {'none', None}:
        return module

    else:
        raise ValueError(f'Unsupported weight normalization: {norm}')


def get_pooling_layer(dimensions: int,
                      pooling_type: str,
                      kernel_size: int = 2,
                      ) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_pool = getattr(nn, class_name)
    return class_pool(kernel_size)


def get_activation(act: str,
                   dim: int = 1,
                   ) -> Optional[nn.Module]:
    if act == 'relu':
        module = nn.ReLU(inplace=True)
    elif act == 'lrelu':
        module = nn.LeakyReLU(0.2, inplace=True)
    elif act == 'prelu':
        module = nn.PReLU()
    elif act == 'elu':
        module = nn.ELU(inplace=True)
    elif act == 'tanh':
        module = nn.Tanh()
    elif act == 'sigmoid':
        module = nn.Sigmoid()
    elif act == 'log_sigmoid':
        module = nn.LogSigmoid()
    elif act == 'softplus':
        module = nn.Softplus()
    elif act == 'softmax':
        module = nn.Softmax(dim)
    elif act == 'log_softmax':
        module = nn.LogSoftmax(dim)
    elif act == 'none':
        module = None
    else:
        raise ValueError(f'Unsupported act: {act}')

    return module


class ConvBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 dilation: int = 1,
                 pad_type: str = 'zero',
                 norm: str = 'none',
                 activation: str = 'relu',
                 use_bias: bool = True,
                 conv_dim: int = 2,
                 weight_norm: str = 'none'
                 ):
        super().__init__()
        self.pad = get_padding_layer(pad_type, padding, conv_dim)
        # TODO padding is redundant here
        assert conv_dim in {1, 2, 3}, f'Unsupported conv_dim={conv_dim}'
        if conv_dim == 1:
            conv = nn.Conv1d
        elif conv_dim == 2:
            conv = nn.Conv2d
        elif conv_dim == 3:
            conv = nn.Conv3d

        self.conv = conv(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=use_bias)
        self.conv = set_weight_normalization(self.conv, norm=weight_norm)

        self.norm_type = norm
        self.norm = get_norm_layer(norm, output_dim, conv_dim)

        self.activation = get_activation(activation, 1)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpliFeatExtractor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            norm: str = 'ln',
            activation: str = 'relu',
            weight_norm: str = 'none',
            concat_x_to_output: bool = True,
            half_size_output: bool = False,
            upsample_mode: str = 'bilinear',
            kernel_size: int = 3,
    ):
        super().__init__()
        self.concat_x_to_output = concat_x_to_output
        self.half_size_output = half_size_output

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.half_size_output:
            self.up2 = nn.Upsample(scale_factor=1, mode=upsample_mode)
            self.up3 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up4 = nn.Upsample(scale_factor=4, mode=upsample_mode)
            self.dw_input = nn.Upsample(scale_factor=0.5, mode=upsample_mode)
        else:
            self.up2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up3 = nn.Upsample(scale_factor=4, mode=upsample_mode)
            self.up4 = nn.Upsample(scale_factor=8, mode=upsample_mode)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # self.conv1a = ConvBlock(input_dim, c1, kernel_size=3, stride=1, padding=1,
        #                         norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv1a = ConvBlock(input_dim, c1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv1b = ConvBlock(c1, c1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv2a = ConvBlock(c1, c2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv2b = ConvBlock(c2, c2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv3a = ConvBlock(c2, c3, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv3b = ConvBlock(c3, c3, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv4a = ConvBlock(c3, c4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.conv4b = ConvBlock(c4, c4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)

        self.convDa = ConvBlock(c1 + c2 + c3 + c4, c5, kernel_size=3, stride=1, padding=1,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)
        self.convDb = ConvBlock(c5, output_dim, kernel_size=1, stride=1, padding=0,
                                norm=norm, activation=activation, conv_dim=2, weight_norm=weight_norm)

    def forward(self, x_input):
        x = self.conv1a(x_input)
        x1 = self.conv1b(x)
        x = self.pool(x1)
        if self.half_size_output:
            x1 = x
        x = self.conv2a(x)
        x2 = self.conv2b(x)
        x = self.pool(x2)
        x = self.conv3a(x)
        x3 = self.conv3b(x)
        x = self.pool(x3)
        x = self.conv4a(x)
        x4 = self.conv4b(x)

        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = self.up4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        cDa = self.convDa(x)
        descriptors = self.convDb(cDa)
        if self.concat_x_to_output:
            if self.half_size_output:
                x_input = self.dw_input(x_input)
            return torch.cat([descriptors, x_input], dim=1)
        return descriptors