import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, mode: str = "nearest"):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv    
        self.upsample = upsample
        self.mode = mode

        if self.upsample and self.mode == "fourier_ap":
            self.fourier_up = ProposedFourierUpsampling(channels=dim_in)
        else:
            self.fourier_up = None
        
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        #self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        #self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(dim_in, dim_out, 3, 1, 0))
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(dim_out, dim_out, 3, 1, 0))
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _upsample(self, x):
        if not self.upsample:
            return x
        if self.mode == "fourier_ap":
            return self.fourier_up(x)
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

    def _shortcut(self, x):
        if self.upsample:
            x = self.fourier_up(x)
            #x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self._upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=384, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        # dim_in = 2**14 // img_size
        dim_in = 64
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        # repeat_num = int(np.log2(img_size)) - 4
        repeat_num = 4
        # if w_hpf > 0:
        #     repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            # if (masks is not None) and (x.size(2) in [32, 64, 128]):
            #     cache[x.size(2)] = x
            if x.size(2) in [48, 96, 192]:
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            # if (masks is not None) and (x.size(2) in [32, 64, 128]):
            #     mask = masks[0] if x.size(2) in [32] else masks[1]
            #     mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
            #     x = x + self.hpf(mask * cache[x.size(2)])
            if x.size(2) in [48, 96, 192]:
                # mask = masks[0] if x.size(2) in [32] else masks[1]
                # mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                # x = x + self.hpf(mask * cache[x.size(2)])
                x = x + self.hpf(cache[x.size(2)])
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0)))
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=384, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        # dim_in = 2**14 // img_size
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # repeat_num = int(np.log2(img_size)) - 2
        repeat_num = 6
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        #blocks += [nn.PReLU()]
        blocks += [nn.Conv2d(dim_out, dim_out, 6, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        #blocks += [nn.PReLU()]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        #idx = torch.arange(y.size(0), device=y.device)
        idx = torch.LongTensor(range(y.size(0)))
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=384, num_domains=2, max_conv_dim=512):
        super().__init__()
        # dim_in = 2**14 // img_size
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # repeat_num = int(np.log2(img_size)) - 2
        repeat_num = 6
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        #blocks += [nn.PReLU()]
        blocks += [nn.Conv2d(dim_out, dim_out, 6, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        #blocks += [nn.PReLU()]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        #idx = torch.arange(y.size(0), device=y.device)
        idx = torch.LongTensor(range(y.size(0)))
        out = out[idx, y]  # (batch)
        return out


class ProposedFourierUpsampling(nn.Module):
    def __init__(self, channels: int, norm: str = "ortho"):
        super().__init__()
        self.norm = norm
        self.channels = channels
        
        # High-frequency emphasis network
        # For CCM images, we want to enhance edges (nerve fiber) which are high-freq
        self.high_freq_enhancer_real = nn.Conv2d(channels, channels, 1, bias=True)
        self.high_freq_enhancer_imag = nn.Conv2d(channels, channels, 1, bias=True)
        
        # Mid-frequency processing (texture)
        self.mid_freq_processor_real = nn.Conv2d(channels, channels, 1, bias=True)
        self.mid_freq_processor_imag = nn.Conv2d(channels, channels, 1, bias=True)
        
        # Low-frequency (base structure)
        self.low_freq_processor_real = nn.Conv2d(channels, channels, 1, bias=True)
        self.low_freq_processor_imag = nn.Conv2d(channels, channels, 1, bias=True)
        
        # Initialize near identity
        for conv in [self.high_freq_enhancer_real, self.high_freq_enhancer_imag,
                     self.mid_freq_processor_real, self.mid_freq_processor_imag,
                     self.low_freq_processor_real, self.low_freq_processor_imag]:
            nn.init.eye_(conv.weight.squeeze())
            nn.init.zeros_(conv.bias)
        
        # Learnable frequency band weights 
        self.freq_band_weights = nn.Parameter(torch.tensor([0.5, 0.8, 1.2]))
        
        # Edge enhancement factor
        self.edge_scale = nn.Parameter(torch.tensor(1.5))
        
        # Global scale
        self.global_scale = nn.Parameter(torch.ones(1))
    
    def _create_frequency_masks(self, H, W, device):
        """Create frequency band masks optimized for edge detection."""
        fy = torch.fft.fftfreq(H, device=device).view(H, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, W)
        freq_radius = torch.sqrt(fy**2 + fx**2)
        
        # Normalize
        max_freq = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=device))
        freq_radius_norm = freq_radius / (max_freq + 1e-8)
        
        # Create band masks
        # Low: 0-0.25 (DC and nearby - base structure)
        # Mid: 0.25-0.6 (texture)
        # High: 0.6-1.0 (edges, nerve fibers)
        low_mask = (freq_radius_norm < 0.25).float()
        mid_mask = ((freq_radius_norm >= 0.25) & (freq_radius_norm < 0.6)).float()
        high_mask = (freq_radius_norm >= 0.6).float()
        
        return low_mask, mid_mask, high_mask
    
    def _directional_frequency_masks(self, H, W, device):
        """Create directional masks to enhance cracks in all orientations."""
        fy = torch.fft.fftfreq(H, device=device).view(H, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, W)
        
        # Compute angle in frequency domain
        angle = torch.atan2(fy, fx)
        
        # Create masks for different orientations 
        # This helps preserve nerve fibers in all directions
        dir_masks = []
        for target_angle in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
            # Tolerance around each angle
            tolerance = math.pi / 8
            mask = torch.cos(angle - target_angle).abs() > math.cos(tolerance)
            dir_masks.append(mask.float())
        
        # Combine all directional masks
        combined_dir_mask = torch.stack(dir_masks).sum(0).clamp(0, 1)
        
        return combined_dir_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_dtype = x.dtype
        x32 = x.float()
        
        B, C, H, W = x32.shape
        device = x32.device
        
        # Constrain parameters
        global_scale = torch.sigmoid(self.global_scale) * 2.0
        edge_scale = torch.sigmoid(self.edge_scale) * 2.0
        
        # Normalize frequency band weights
        band_weights = F.softmax(self.freq_band_weights, dim=0) * 3.0  
        
        # FFT
        X = torch.fft.fft2(x32, dim=(-2, -1), norm=self.norm)
        
        # Create frequency masks
        low_mask, mid_mask, high_mask = self._create_frequency_masks(H, W, device)
        dir_mask = self._directional_frequency_masks(H, W, device)
        
        # Expand masks for broadcasting
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        mid_mask = mid_mask.unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.unsqueeze(0).unsqueeze(0)
        dir_mask = dir_mask.unsqueeze(0).unsqueeze(0)
        
        # Process each frequency band separately
        # Low frequencies (structure)
        X_real_low = self.low_freq_processor_real(X.real * low_mask) * band_weights[0]
        X_imag_low = self.low_freq_processor_imag(X.imag * low_mask) * band_weights[0]
        
        # Mid frequencies (texture)
        X_real_mid = self.mid_freq_processor_real(X.real * mid_mask) * band_weights[1]
        X_imag_mid = self.mid_freq_processor_imag(X.imag * mid_mask) * band_weights[1]
        
        # High frequencies (edges/nerve fibers) 
        # Apply directional mask to preserve nerve fiber patterns in all orientations
        high_enhanced_real = X.real * high_mask * dir_mask
        high_enhanced_imag = X.imag * high_mask * dir_mask
        
        X_real_high = self.high_freq_enhancer_real(high_enhanced_real) * band_weights[2] * edge_scale
        X_imag_high = self.high_freq_enhancer_imag(high_enhanced_imag) * band_weights[2] * edge_scale
        
        # Combine all bands
        X_real_processed = X_real_low + X_real_mid + X_real_high
        X_imag_processed = X_imag_low + X_imag_mid + X_imag_high
        
        # Add residual connection
        X_real_final = X_real_processed * global_scale + X.real
        X_imag_final = X_imag_processed * global_scale + X.imag
        
        X_transformed = torch.complex(X_real_final, X_imag_final)
        
        # Zero-pad for 2x upsampling
        X_shifted = torch.fft.fftshift(X_transformed, dim=(-2, -1))
        X_padded = F.pad(X_shifted, (W//2, W//2, H//2, H//2), mode='constant', value=0)
        Y = torch.fft.ifftshift(X_padded, dim=(-2, -1))
        
        # Inverse FFT
        y = torch.fft.ifft2(Y, dim=(-2, -1), norm=self.norm).real        

        # Safety check
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: NaN/Inf detected, using fallback")
            return F.interpolate(x, scale_factor=2, mode='nearest')
        
        return y.to(dtype=inp_dtype)