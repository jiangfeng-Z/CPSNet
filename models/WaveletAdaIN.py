import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletAdaIN(nn.Module):
    def __init__(self, wavelet='db1'):
        super().__init__()
        self.dwt = DWTForward(J=2, wave=wavelet)
        self.idwt = DWTInverse(wave=wavelet)

    def adain(self, content, style):
        c_mean = content.mean(dim=[2, 3], keepdim=True)
        c_std = content.std(dim=[2, 3], keepdim=True) + 1e-5
        s_mean = style.mean(dim=[2, 3], keepdim=True)
        s_std = style.std(dim=[2, 3], keepdim=True) + 1e-5
        return (content - c_mean) * (s_std / c_std) + s_mean

    def forward(self, content, style):
        c_low, c_high = self.dwt(content)
        s_low, s_high = self.dwt(style)

        fused_low = self.adain(c_low, s_low)

        fused_high = []
        for c_h_level, s_h_level in zip(c_high, s_high):
            B, C, D, H, W = c_h_level.shape
            fused_dir = []
            for d in range(D):
                fused = self.adain(c_h_level[:, :, d, :, :],
                                   s_h_level[:, :, d, :, :])
                fused_dir.append(fused.unsqueeze(2))
            fused_high.append(torch.cat(fused_dir, dim=2))

        return self.idwt((fused_low, fused_high))