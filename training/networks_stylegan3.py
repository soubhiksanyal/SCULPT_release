# Modified from StyleGAN3 codebase

"""SCULPT generator"""

from copy import deepcopy
import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from training.renderer import NormalRender, SMPL_Layer, displacement_Layer, TextureRender
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
import ipdb

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        assert noise_mode in ['random', 'const', 'none'] # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)
        
        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        # Construct layers.
        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels[prev]), out_channels= int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        disp_activatn_type,         # Activation type after the displacement layer
        mask_disp_map,              # mask the displacement maps with SMPL UV mask
        disp_scale,                 # Scaling factor for the displacements after activation
        only_disp_img,              # If Turned on only displacement image is generated
        resume_pretrain_cape,       # If Turned on resumed from a pre-trained generator on CAPE
        seperate_disp_map,          # If Turned on seperate displacement image for 10 different body parts are generated
        sep_disp_map_sampling,      # If Turned on seperate displacement image for 10 different body parts are generated and sampled by indexing
        spiral_conv,                # If Turned on spiral convolution is performed at the end on the posed mesh
        texture_render,             # If turned on enable texture rendering 
        guass_blur_normals,         # If turned on blur the normal renderings
        conformnet,                 # If turned on enable conditioning texture network with geometry network 
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        render_kwargs       = {},   # Arguments for Rendering Pipeline.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim

        img_channels = 3

        self.conformnet = conformnet

        self.seperate_disp_map = seperate_disp_map
        self.sep_disp_map_sampling = sep_disp_map_sampling

        self.spiral_conv = spiral_conv
        self.texture_render = texture_render
        self.guass_blur_normals = guass_blur_normals

        if self.seperate_disp_map:
            img_channels = 10 * 3

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        if self.texture_render:
            self.TextureRender = TextureRender(img_size=img_resolution, **render_kwargs)
        else:
            self.Normalrender = NormalRender(img_size=img_resolution, **render_kwargs)

        if self.conformnet:
            self.Normalrender = NormalRender(img_size=img_resolution, background_color=1., **render_kwargs)

        if self.guass_blur_normals:
            self.gauss_blur = GaussianBlur((5,5),(0.01, 1.0))
        # # ipdb.set_trace()
        self.mask_disp_map = mask_disp_map
        self.smpl_body = SMPL_Layer(**render_kwargs)
        self.displacement_Layer = displacement_Layer(img_resolution, **render_kwargs)

        self.only_disp_img = only_disp_img
        self.resume_pretrain_cape= resume_pretrain_cape

        # head_vertices_temp = np.load('../data/smpl_head_verts_indices.npy')
        # hand_vertices_temp = np.load('../data/smpl_hand_feet_verts_indices.npy')
        # head_vertices = np.hstack((head_vertices_temp, hand_vertices_temp))

        # head_verts_mask = torch.ones(6890)
        # head_verts_mask[head_vertices] = 0.0
        # self.register_buffer('head_verts_mask', head_verts_mask)

        # # Load part masks

        # part_0 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_0.png')[:, :, :3]
        # part_1 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_1.png')[:, :, :3]
        # part_2 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_2.png')[:, :, :3]
        # part_3 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_3.png')[:, :, :3]
        # part_4 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_4.png')[:, :, :3]
        # part_5 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_5.png')[:, :, :3]
        # part_6 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_6.png')[:, :, :3]
        # part_7 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_7.png')[:, :, :3]
        # part_8 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_8.png')[:, :, :3]
        # part_9 = plt.imread('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/smpl_parts_256/part_9.png')[:, :, :3]

        # parts = torch.from_numpy(np.concatenate((part_0, part_1, part_2, part_3, part_4, part_5, part_6, part_7, part_8, part_9), 2))

        # self.register_buffer('smpl_uv_parts', parts)

        # parts_indexing = torch.from_numpy(np.load('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/vertex2channel_indexlist_2048.npy')).type(torch.int32)
        # parts_indexing_repeate = torch.repeat_interleave(parts_indexing, 3).reshape(-1,3)
        # parts_indexing_repeate_index = 3*parts_indexing_repeate + ((torch.Tensor([[0,1,2]]).type(torch.int32)) + torch.arange(6890).unsqueeze(1)*30)
        
        # self.register_buffer('parts_indexing_repeate_index', parts_indexing_repeate_index.type(torch.int32))

        self.disp_scale = disp_scale

        smpl_uv_mask = np.load(render_kwargs.SMPL_uv_mask_path)
        smpl_uv_mask2 = deepcopy(smpl_uv_mask)
        smpl_uv_mask2 = smpl_uv_mask - 1
        # ipdb.set_trace()
        self.register_buffer('smpl_uv_mask', torch.from_numpy(smpl_uv_mask))
        self.register_buffer('smpl_uv_mask2', torch.from_numpy(smpl_uv_mask2))
        if disp_activatn_type=='tanh':
            self.threshold_layer = torch.nn.Tanh() # torch.nn.Sigmoid() # 
        else:
            self.threshold_layer = torch.nn.Sigmoid()

        # if self.spiral_conv:
        #     spiral_list = torch.tensor(np.load('/is/cluster/work/ssanyal/project_4/data/stylegan3/smpl_uv_mask/spiralnetPlusPlus_smpl_spirallistfrom_SMPLUV.npy'))
        #     self.spiral_conv_layer_1  = SpiralConv(3,3,spiral_list)
        #     self.spiral_conv_layer_2  = SpiralConv(3,3,spiral_list)



    def forward(self, z, c, body_shape, body_pose, body_cam, G_geometry=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        if self.conformnet:
            return self.forward_conform(z, c, body_shape, body_pose, body_cam, G_geometry=G_geometry, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas, **synthesis_kwargs)
        else:
            return self.forward_sep(z, c, body_shape, body_pose, body_cam, G_geometry=G_geometry, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas, **synthesis_kwargs)


    def forward_conform(self, z, c, body_shape, body_pose, body_cam, G_geometry=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        if c.shape[1]==6:
            ws_geo = G_geometry.mapping(z, torch.cat((c, body_pose[:,3:66]),1), truncation_psi=1)
        elif c.shape[1]==518:
            ws_geo = G_geometry.mapping(z, torch.cat((c[:, :6], body_pose[:,3:66]),1), truncation_psi=1)

        ## Texture Network
        misc.assert_shape(ws, [None, self.synthesis.num_ws, self.synthesis.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)
        ws_geo = ws_geo.to(torch.float32).unbind(dim=1)
        # ipdb.set_trace()
        # Execute layers.
        x = self.synthesis.input(ws[0])
        x_geo = G_geometry.synthesis.input(ws_geo[0])
        for name, w, w_geo in zip(self.synthesis.layer_names, ws[1:], ws_geo[1:]):
            # print(name)
            # if name.split('_')[0] == 'L0' or name.split('_')[0] == 'L1' or name.split('_')[0] == 'L2' or \
            #     name.split('_')[0] == 'L5' or name.split('_')[0] == 'L6' or name.split('_')[0] == 'L7' or \
            #         name.split('_')[0] == 'L8' or name.split('_')[0] == 'L9':

            x = x + x_geo
            x_geo = getattr(G_geometry.synthesis, name)(x_geo, w_geo, update_emas=False, **synthesis_kwargs)
            x = getattr(self.synthesis, name)(x, w, update_emas=update_emas, **synthesis_kwargs)
            # print('x->',x.shape)
            # print('x_geo->',x.shape)
        if self.synthesis.output_scale != 1:
            x = x * self.synthesis.output_scale

        if G_geometry.synthesis.output_scale != 1:
            x_geo = x_geo * G_geometry.synthesis.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x_geo, [None, G_geometry.synthesis.img_channels, G_geometry.synthesis.img_resolution, G_geometry.synthesis.img_resolution])
        UV_geo = x_geo.to(torch.float32)

        misc.assert_shape(x, [None, self.synthesis.img_channels, self.synthesis.img_resolution, self.synthesis.img_resolution])
        UV_tex = x.to(torch.float32)


        disp_img_geo = (UV_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
        vert_disps = G_geometry.displacement_Layer(disp_img_geo)

        disp_img_out = UV_tex * 1.0

        clothed_body_shape = body_shape + vert_disps
        posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))

        img, mesh = self.TextureRender(posed_body, body_cam=body_cam, text_img=UV_tex.permute(0,2,3,1))

        norm_img, _ = self.Normalrender(posed_body, body_cam=body_cam)

        return img.permute(0,3,1,2), mesh, torch.cat((disp_img_out, norm_img.permute(0,3,1,2)[:,:3,:,:]),1)


    def forward_sep(self, z, c, body_shape, body_pose, body_cam, G_geometry=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        disp_img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        disp_img_out = disp_img * 1.0

        # if self.seperate_disp_map:

        #     if not self.sep_disp_map_sampling:
        #         disp_img = disp_img * self.smpl_uv_parts.unsqueeze(0).permute(0,3,1,2)

        #     disp_img_2 = disp_img * self.smpl_uv_parts.unsqueeze(0).permute(0,3,1,2)

        #     disp_img_out = disp_img_2[:, :3, :, :] + disp_img_2[:, 3:6, :, :] + disp_img_2[:, 6:9, :, :] + disp_img_2[:,9:12, :, :] \
        #                     + disp_img_2[:, 12:15, :, :] + disp_img_2[:, 15:18, :, :] + disp_img_2[:, 18:21, :, :] + disp_img_2[:, 21:24, :, :] \
        #                     + disp_img_2[:, 24:27, :, :] + disp_img_2[:, 27:, :, :] \
        #                     + (self.smpl_uv_mask2.unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1)

        #     if self.only_disp_img:

        #         disp_img = disp_img[:, :3, :, :] + disp_img[:, 3:6, :, :] + disp_img[:, 6:9, :, :] + disp_img[:,9:12, :, :] \
        #                         + disp_img[:, 12:15, :, :] + disp_img[:, 15:18, :, :] + disp_img[:, 18:21, :, :] + disp_img[:, 21:24, :, :] \
        #                         + disp_img[:, 24:27, :, :] + disp_img[:, 27:, :, :]

        if self.mask_disp_map:
            disp_img = disp_img * (self.smpl_uv_mask.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1) \
                + (self.smpl_uv_mask2.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1)
            # disp_img_out = disp_img * 1.0
        if self.only_disp_img:
            if self.seperate_disp_map:
                return disp_img, 0., disp_img_out
            return disp_img, 0., disp_img_out
        # # verts = self.smpl_body(torch.zeros(3, 6890, 3).to('cuda').type(torch.float32), torch.zeros(3, 69).to('cuda').type(torch.float32), torch.zeros(3, 3).to('cuda').type(torch.float32))
        # # normal_img = self.Normalrender(verts).permute(0.3,1,2)
        # vert_disps = self.displacement_Layer(disp_img)
        # clothed_body_shape = body_shape + vert_disps
        # posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:], body_pose[:, :3])
        # img = self.Normalrender(clothed_body_shape).permute(0,3,1,2)
        # ipdb.set_trace()
        
        if self.texture_render:
            if c.shape[1] == G_geometry.c_dim:
                ws_geo = G_geometry.mapping(z, c, truncation_psi=1)
            else:
                ws_geo = G_geometry.mapping(z, torch.cat((c, body_pose[:,3:66]),1), truncation_psi=1)
            disp_img_geo = G_geometry.synthesis(ws_geo, noise_mode='const')
            # disp_img_geo, _ = G_geometry(z, c, body_shape, body_pose, body_cam, truncation_psi=1, noise_mode='const')
            disp_img_geo = (disp_img_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
            vert_disps = G_geometry.displacement_Layer(disp_img_geo)
            # clothed_body_shape = body_shape + vert_disps_all
            # posed_body = G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3])

        else:
            if self.resume_pretrain_cape:
                disp_img = (disp_img * 0.5 + 0.5) * 2 * 0.071 - 0.071
                # vert_disps = self.displacement_Layer(disp_img)
                vert_disps_all = self.displacement_Layer(disp_img) * self.head_verts_mask[:, None]
                # ipdb.set_trace()
                if self.img_channels == 30:
                    if self.sep_disp_map_sampling:
                            # ipdb.set_trace()
                            pass # TODO: Needs implementation
                            # vert_disps = torch.index_select(vert_disps_all.reshape(body_shape.shape[0],-1), -1, self.parts_indexing_repeate_index.view(-1)).reshape(body_shape.shape[0], 6890,-1)
                            # ipdb.set_trace()
                    else:
                        vert_disps = vert_disps_all[:, :, :3] + vert_disps_all[:, :, 3:6] + vert_disps_all[:, :, 6:9] + vert_disps_all[:, :, 9:12] \
                                        + vert_disps_all[:, :, 12:15] + vert_disps_all[:, :, 15:18] + vert_disps_all[:, :, 18:21] + vert_disps_all[:, :, 21:24] \
                                        + vert_disps_all[:, :, 24:27] + vert_disps_all[:, :, 27:]
                else:
                    vert_disps = vert_disps_all
            else:
                if self.img_channels == 3:
                    vert_disps = self.displacement_Layer(disp_img) * self.head_verts_mask[:, None] * self.disp_scale # self.threshold_layer(self.displacement_Layer(disp_img) * self.head_verts_mask[:, None]) * self.disp_scale # self.threshold_layer(self.displacement_Layer(disp_img)) * self.disp_scale # self.displacement_Layer(disp_img) #
                elif self.img_channels == 9:
                    vert_disps_all = self.threshold_layer(self.displacement_Layer(disp_img))
                    vert_disps = vert_disps_all[:, :, :3] * self.disp_scale + vert_disps_all[:, :, 3:6] * self.disp_scale * 0.5 \
                                    + vert_disps_all[:, :, 6:] * self.disp_scale * 0.25
                    vert_disps = vert_disps * self.head_verts_mask[:, None]

                elif self.img_channels == 30:
                    vert_disps_all = self.displacement_Layer(disp_img) * self.head_verts_mask[:, None] * self.disp_scale

                    if self.sep_disp_map_sampling:
                        # ipdb.set_trace()
                        pass # TODO: Needs implementation
                        # vert_disps = torch.index_select(vert_disps_all.reshape(body_shape.shape[0],-1), -1, self.parts_indexing_repeate_index.view(-1)).reshape(body_shape.shape[0], 6890,-1)
                        # ipdb.set_trace()
                    else:
                        vert_disps = vert_disps_all[:, :, :3] + vert_disps_all[:, :, 3:6] + vert_disps_all[:, :, 6:9] + vert_disps_all[:, :, 9:12] \
                                        + vert_disps_all[:, :, 12:15] + vert_disps_all[:, :, 15:18] + vert_disps_all[:, :, 18:21] + vert_disps_all[:, :, 21:24] \
                                        + vert_disps_all[:, :, 24:27] + vert_disps_all[:, :, 27:]
                    
            # # ipdb.set_trace()
            # if self.spiral_conv:
            #     vert_disps = vert_disps + self.spiral_conv_layer_1(vert_disps)
            #     vert_disps = vert_disps + self.spiral_conv_layer_2(vert_disps)
    
        clothed_body_shape = body_shape + vert_disps#.unsqueeze(0).repeat(body_shape.shape[0], 1, 1)
        # posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:])
        # posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3])
        posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))
        # ipdb.set_trace()
        if self.texture_render:
            img, mesh = self.TextureRender(posed_body, body_cam=body_cam, text_img=disp_img.permute(0,2,3,1))
            # img = img * 2.0 - 1.0
        else:
            img, mesh = self.Normalrender(posed_body, body_cam=body_cam)

        # ipdb.set_trace()
        return img.permute(0,3,1,2), mesh, disp_img_out

#----------------------------------------------------------------------------


# @persistence.persistent_class
# class Generator_onlydisps(torch.nn.Module):
#     def __init__(self,
#         z_dim,                      # Input latent (Z) dimensionality.
#         c_dim,                      # Conditioning label (C) dimensionality.
#         w_dim,                      # Intermediate latent (W) dimensionality.
#         img_resolution,             # Output resolution.
#         img_channels,               # Number of output color channels.
#         mask_disp_map,              # mask the displacement maps with SMPL UV mask
#         mapping_kwargs      = {},   # Arguments for MappingNetwork.
#         render_kwargs       = {},   # Arguments for Rendering Pipeline.
#         **synthesis_kwargs,         # Arguments for SynthesisNetwork.
#     ):
#         super().__init__()
#         self.z_dim = z_dim
#         self.c_dim = c_dim
#         self.w_dim = w_dim
#         self.img_resolution = img_resolution
#         self.img_channels = img_channels
#         # self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
#         # self.num_ws = self.synthesis.num_ws
#         # self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
#         self.Normalrender = NormalRender(img_size=img_resolution, **render_kwargs)
#         # ipdb.set_trace()
#         self.mask_disp_map = mask_disp_map
#         self.smpl_body = SMPL_Layer(**render_kwargs)
#         self.displacement_Layer = displacement_Layer(img_resolution, **render_kwargs)

#         optimisable_disp_verts = torch.full((6890, 3), 0.0) # torch.randn(6890, 3) * 0.01 #
#         self.disp_verts = torch.nn.Parameter(optimisable_disp_verts)

#         head_vertices_temp = np.load('/ps/project/tag_3d/data/smpl/smpl_parts/smpl_head_verts_indices.npy')
#         hand_vertices_temp = np.load('/ps/project/tag_3d/data/smpl/smpl_parts/smpl_hand_feet_verts_indices.npy')
#         head_vertices = np.hstack((head_vertices_temp, hand_vertices_temp))

#         head_verts_mask = torch.ones_like(self.disp_verts[:, 0])
#         head_verts_mask[head_vertices] = 0.0
#         self.register_buffer('head_verts_mask', head_verts_mask)

#         smpl_uv_mask = np.load(render_kwargs.SMPL_uv_mask_path)
#         smpl_uv_mask2 = deepcopy(smpl_uv_mask)
#         smpl_uv_mask2 = smpl_uv_mask - 1
#         # ipdb.set_trace()
#         self.register_buffer('smpl_uv_mask', torch.from_numpy(smpl_uv_mask))
#         self.register_buffer('smpl_uv_mask2', torch.from_numpy(smpl_uv_mask2))
#         self.threshold_layer = torch.nn.Sigmoid() #torch.nn.Tanh()



#     def forward(self, z, c, body_shape, body_pose, body_cam, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
#         # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
#         # disp_img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

#         # if self.mask_disp_map:
#         #     disp_img = disp_img * (self.smpl_uv_mask.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1) \
#         #         + (self.smpl_uv_mask2.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1)

#         # # verts = self.smpl_body(torch.zeros(3, 6890, 3).to('cuda').type(torch.float32), torch.zeros(3, 69).to('cuda').type(torch.float32), torch.zeros(3, 3).to('cuda').type(torch.float32))
#         # # normal_img = self.Normalrender(verts).permute(0.3,1,2)
#         # ipdb.set_trace()
#         vert_disps = self.threshold_layer(self.disp_verts * self.head_verts_mask[:, None]) * 0.1 - 0.05 #self.displacement_Layer(disp_img)
#         clothed_body_shape = body_shape + vert_disps.unsqueeze(0).repeat(body_shape.shape[0], 1, 1)
#         posed_body = self.smpl_body(clothed_body_shape, body_pose[:, 3:], body_pose[:, :3])
#         img, mesh = self.Normalrender(posed_body, body_cam=body_cam) #img, _ = self.Normalrender(clothed_body_shape)
#         # ipdb.set_trace()
#         return img.permute(0,3,1,2), mesh

# #----------------------------------------------------------------------------

@persistence.persistent_class
class SpiralConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super().__init__()
        self.SpiralConv_dim = dim
        # self.SpiralConv_indices = indices
        self.register_buffer('SpiralConv_indices', indices)
        self.SpiralConv_in_channels = in_channels
        self.SpiralConv_out_channels = out_channels
        self.SpiralConv_seq_length = indices.size(1)

        self.SpiralConv_layer = torch.nn.Linear(in_channels * self.SpiralConv_seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.SpiralConv_layer.weight, gain=0.001)
        torch.nn.init.constant_(self.SpiralConv_layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.SpiralConv_indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.SpiralConv_indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.SpiralConv_dim, self.SpiralConv_indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.SpiralConv_layer(x)
        return x

    def extra_repr(self):
        return 'SpiralConv_in_channels={}, SpiralConv_out_channels={}, {}, seq_length={}'.format(self.__class__.__name__,
                                                  self.SpiralConv_in_channels,
                                                  self.SpiralConv_out_channels,
                                                  self.SpiralConv_seq_length)
