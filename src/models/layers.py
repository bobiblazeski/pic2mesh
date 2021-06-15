import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    This layer can also be used as filtering layer by setting `scale_factor` as 1.
    """

    def __init__(self,
                 scale_factor=2,
                 kernel=(1, 3, 3, 1),
                 extra_padding=0,
                 kernel_gain=None):
        super().__init__()
        assert scale_factor >= 1
        self.scale_factor = scale_factor

        if extra_padding != 0:
            assert scale_factor == 1

        if kernel is None:
            kernel = np.ones((scale_factor), dtype=np.float32)
        else:
            kernel = np.array(kernel, dtype=np.float32)
        assert kernel.ndim == 1
        kernel = np.outer(kernel, kernel)
        kernel = kernel / np.sum(kernel)
        if kernel_gain is None:
            kernel = kernel * (scale_factor ** 2)
        else:
            assert kernel_gain > 0
            kernel = kernel * (kernel_gain ** 2)
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.kernel = self.kernel.flip(0, 1)

        self.upsample_padding = (0, scale_factor - 1,  # Width padding.
                                 0, 0,  # Width.
                                 0, scale_factor - 1,  # Height padding.
                                 0, 0,  # Height.
                                 0, 0,  # Channel.
                                 0, 0)  # Batch size.

        padding = kernel.shape[2] - scale_factor + extra_padding
        self.padding = ((padding + 1) // 2 + scale_factor - 1, padding // 2,
                        (padding + 1) // 2 + scale_factor - 1, padding // 2)

    def forward(self, x):
        assert len(x.shape) == 4
        channels = x.shape[1]
        if self.scale_factor > 1:
            x = x.view(-1, channels, x.shape[2], 1, x.shape[3], 1)
            x = F.pad(x, self.upsample_padding, mode='constant', value=0)
            x = x.view(-1, channels, x.shape[2] * self.scale_factor,
                       x.shape[4] * self.scale_factor)
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        x = F.pad(x, self.padding, mode='constant', value=0)
        x = F.conv2d(x, self.kernel, stride=1)
        x = x.view(-1, channels, x.shape[2], x.shape[3])
        return x

class DenseBlock(nn.Module):
    """Implements the dense block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_gain=1.0,
                 lr_multiplier=0.01,
                 add_bias=True,
                 init_bias=0,
                 activation_type='lrelu'):
        """Initializes the class with block settings.

        NOTE: Wscale is used as default.

        Args:
          in_channels: Number of channels of the input tensor.
          out_channels: Number of channels of the output tensor.
          weight_gain: Gain factor for weight parameter in dense layer.
          lr_multiplier: Learning rate multiplier.
          add_bias: Whether to add bias after fully-connected operation.
          init_bias: Initialized bias.
          activation_type: Type of activation function. Support `linear`, `relu`
            and `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()

        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels,
                            bias=False)

        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.init_bias = init_bias

        self.weight_scale = weight_gain / np.sqrt(in_channels)
        self.lr_multiplier = lr_multiplier

        if activation_type == 'linear':
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == 'relu':
            self.activate = nn.ReLU(inplace=True)
            self.activate_scale = np.sqrt(2.0)
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'{activation_type}!')

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.view(x.shape[0], -1)
        x = self.fc(x) * self.weight_scale * self.lr_multiplier
        if self.add_bias:
            x = x + self.bias.view(1, -1) * self.lr_multiplier + self.init_bias
        x = self.activate(x) * self.activate_scale
        return x

class ModulateConvBlock(nn.Module):
    """Implements the convolutional block with style modulation."""

    def __init__(self,                 
                 w_space_dim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 scale_factor=1,
                 filtering_kernel=(1, 3, 3, 1),                 
                 fused_modulate=True,
                 demodulate=True,
                 weight_gain=1.0,
                 lr_multiplier=1.0,
                 add_bias=True,
                 activation_type='lrelu',
                 add_noise=True,                 
                 epsilon=1e-8):
        """Initializes the class with block settings.

        NOTE: Wscale is used as default.

        Args:
          w_space_dim: Dimension of disentangled latent space. This is used for
            style modulation.
          in_channels: Number of channels of the input tensor.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          scale_factor: Scale factor for upsampling. `1` means skip upsampling.
          filtering_kernel: Kernel used for filtering after upsampling.          
          fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
          demodulate: Whether to perform style demodulation.
          weight_gain: Gain factor for weight parameter in convolutional layer.
          lr_multiplier: Learning rate multiplier.
          add_bias: Whether to add bias after convolution.
          activation_type: Type of activation function. Support `linear`, `relu`
            and `lrelu`.
          add_noise: Whether to add noise to spatial feature map.
          randomize_noise: Whether to randomize new noises at runtime.
          epsilon: Small number to avoid `divide by zero`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()
        
        self.in_c = in_channels
        self.out_c = out_channels
        self.ksize = kernel_size
        self.eps = epsilon

        self.weight = nn.Parameter(
            torch.randn(kernel_size, kernel_size, in_channels, out_channels))
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_scale = weight_gain / np.sqrt(fan_in)
        self.lr_multiplier = lr_multiplier

        self.scale_factor = scale_factor
        if scale_factor > 1:
            self.filter = UpsamplingLayer(scale_factor=1,
                                          kernel=filtering_kernel,
                                          extra_padding=scale_factor - kernel_size,
                                          kernel_gain=scale_factor)
        else:
            assert kernel_size % 2 == 1
            self.conv_padding = kernel_size // 2

        self.w_space_dim = w_space_dim
        self.style = DenseBlock(in_channels=w_space_dim,
                                out_channels=in_channels,
                                lr_multiplier=1.0,
                                init_bias=1.0,
                                activation_type='linear')

        self.fused_modulate = fused_modulate
        self.demodulate = demodulate

        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        if activation_type == 'linear':
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == 'relu':
            self.activate = nn.ReLU(inplace=True)
            self.activate_scale = np.sqrt(2.0)
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'{activation_type}!')

        self.add_noise = add_noise        
        if add_noise:            
            self.noise_strength = nn.Parameter(torch.zeros(()))

    def get_resolution(self, x, batch, out_c):
        return int(math.sqrt((x.numel() // batch) // out_c))
    
    def forward(self, x, w):
        assert (len(x.shape) == 4 and len(w.shape) == 2 and
                w.shape[0] == x.shape[0] and w.shape[1] == self.w_space_dim)
        batch = x.shape[0]

        weight = self.weight * self.weight_scale * self.lr_multiplier

        # Style modulation.
        style = self.style(w)
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * style.view(batch, 1, 1, self.in_c, 1)

        # Style demodulation.
        if self.demodulate:
            _weight_norm = torch.sqrt(
                torch.sum(_weight ** 2, dim=[1, 2, 3]) + self.eps)
            _weight = _weight / _weight_norm.view(batch, 1, 1, 1, self.out_c)

        if self.fused_modulate:
            x = x.view(1, batch * self.in_c, x.shape[2], x.shape[3])
            weight = _weight.permute(1, 2, 3, 0, 4).reshape(
                self.ksize, self.ksize, self.in_c, batch * self.out_c)
        else:
            x = x * style.view(batch, self.in_c, 1, 1)

        if self.scale_factor > 1:
            weight = weight.flip(0, 1)
            if self.fused_modulate:
                weight = weight.view(
                    self.ksize, self.ksize, self.in_c, batch, self.out_c)
                weight = weight.permute(0, 1, 4, 3, 2)
                weight = weight.reshape(
                    self.ksize, self.ksize, self.out_c, batch * self.in_c)
                weight = weight.permute(3, 2, 0, 1)
            else:
                weight = weight.permute(2, 3, 0, 1)
            x = F.conv_transpose2d(x, weight, stride=self.scale_factor, padding=0,
                                   groups=(batch if self.fused_modulate else 1))
            x = self.filter(x)
        else:
            weight = weight.permute(3, 2, 0, 1)
            x = F.conv2d(x, weight, stride=1, padding=self.conv_padding,
                         groups=(batch if self.fused_modulate else 1))
        
        res  = self.get_resolution(x, batch, self.out_c)
        if self.fused_modulate:
            x = x.view(batch, self.out_c, res, res)
        elif self.demodulate:
            x = x / _weight_norm.view(batch, self.out_c, 1, 1)

        if self.add_noise:
            noise = torch.randn(x.shape[0], 1, res, res).to(x)
            x = x + noise * self.noise_strength.view(1, 1, 1, 1)

        if self.add_bias:
            bias = self.bias * self.lr_multiplier
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale

        return x
