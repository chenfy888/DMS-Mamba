import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import numpy as np
import logging
import os
from mamba_ssm.modules.mamba_simple import Mamba
from .pan_mamba_simple import Mamba
from .pan_refine import Refine
from .ESDR import EDSR, ResBlock
from .fusion_module import Fusion_dynamic
import pywt
from typing import Union, Tuple, Sequence
from .waveelt_block import DWT, IDWT, get_filter_tensors, _as_wavelet
import torch.fft



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
    return x

def fftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2]//2, x.shape[-1]//2), dims=(-2, -1))
    return x

def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2]+1)//2, (x.shape[-1]+1)//2), dims=(-2, -1))
    return x

# ==================== LayerNorm====================
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)



class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0, bias=True)  # 1x1
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        if self.use_HIN:
            out_1, out_2 = torch.chunk(resi, 2, dim=1)
            resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi



class MambaBlock(nn.Module):
    def __init__(self, dim):
        super(MambaBlock, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.norm = LayerNorm(dim, 'with_bias')

    def forward(self, x):
        if isinstance(x, tuple):
            input_tensor, residual = x
            is_tuple_mode = True
        else:
            input_tensor = x
            residual = torch.zeros_like(x)
            is_tuple_mode = False

        residual = input_tensor + residual 
        
        x_norm = self.norm(residual)

        b, c, h, w = x_norm.shape
        x_flat = to_3d(x_norm)  # [B, H*W, C]
        
        mamba_out = self.encoder(x_flat)  # [B, L, D] → [B, L, D]
        
        x_out = to_4d(mamba_out, h, w)  # [B, C, H, W]
        
        if is_tuple_mode:
            return (x_out, residual)
        else:
            return x_out + residual



class LearnableFusion(nn.Module):

    def __init__(self, num_feat):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=1) 
        out = self.fusion_conv(concat)  
        return out


class HighFreqGuidedMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.high_proj = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )
        
        self.norm_low = LayerNorm(dim, 'with_bias')
        self.norm_high = LayerNorm(dim, 'with_bias')
        self.mamba = Mamba(dim, bimamba_type="v3")
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, low, high_cat):
        b, c, h, w = low.shape
        
        high = self.high_proj(high_cat) 
        
        low_tokens = to_3d(low)       
        high_tokens = to_3d(high)
        
        low_norm = self.norm_low(low_tokens) 
        high_norm = self.norm_high(high_tokens)
        
        out_tokens = self.mamba(low_norm, extra_emb=high_norm)
        out_tokens = out_tokens + low_tokens  
        out = to_4d(out_tokens, h, w)
        out = self.dwconv(out) + out
        
        return out


class FRBMamba(nn.Module):
    def __init__(self, dim):
        super(FRBMamba, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3")
        self.norm_t2 = LayerNorm(dim, 'with_bias')
        self.norm_t1 = LayerNorm(dim, 'with_bias')
      
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, t2, t1):
        B, C, H, W = t2.shape
        
        t2_flat = to_3d(t2)
        t1_flat = to_3d(t1)
        
        t2_norm = self.norm_t2(t2_flat)
        t1_norm = self.norm_t1(t1_flat)
        
        out_mamba = self.cross_mamba(t2_norm, extra_emb=t1_norm)
        
        #  (x + Mamba(Norm(x)))
        out = out_mamba + t2_flat
        
        out_4d = to_4d(out, H, W)
        out_4d = self.dwconv(out_4d) + out_4d
        
        return out_4d

# SFDI
class DualDomainCrossMamba(nn.Module):
    def __init__(self, dim, use_fftshift=False):
        super().__init__()
        self.dim = dim
        self.use_fftshift = use_fftshift

        # spatial branch
        self.s_dw_tar = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.s_dw_ref = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.s_norm_tar = LayerNorm(dim, 'with_bias')
        self.s_norm_ref = LayerNorm(dim, 'with_bias')
        self.s_mamba = Mamba(dim, bimamba_type="v3")
        self.s_out_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # freq branch: complex->(re,im) 2C->C
        self.f_in = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.f_norm_tar = LayerNorm(dim, 'with_bias')
        self.f_norm_ref = LayerNorm(dim, 'with_bias')
        self.f_mamba = Mamba(dim, bimamba_type="v3")
        self.f_out = nn.Conv2d(dim, dim * 2, 1, 1, 0)  # C->2C

        self.fuse = LearnableFusion(dim)

    def forward(self, itar, iref):
        B, C, H, W = itar.shape

        # ========= spatial =========
        tar_s = self.s_dw_tar(itar)
        ref_s = self.s_dw_ref(iref)

        tar_tok = self.s_norm_tar(to_3d(tar_s))
        ref_tok = self.s_norm_ref(to_3d(ref_s))

        out_s_tok = self.s_mamba(tar_tok, extra_emb=ref_tok)
        out_s_tok = out_s_tok + to_3d(itar)
        out_s = to_4d(out_s_tok, H, W)
        out_s = self.s_out_dw(out_s) + out_s

        # ========= freq =========
        Tar_F = fft2(itar) 
        Ref_F = fft2(iref)

        if self.use_fftshift:
            Tar_F = fftshift2(Tar_F)
            Ref_F = fftshift2(Ref_F)

        tar_f2 = torch.cat([Tar_F.real, Tar_F.imag], dim=1)  # [B,2C,H,W]
        ref_f2 = torch.cat([Ref_F.real, Ref_F.imag], dim=1)

        tar_f = self.f_in(tar_f2)  # [B,C,H,W]
        ref_f = self.f_in(ref_f2)

        tar_f_tok = self.f_norm_tar(to_3d(tar_f))
        ref_f_tok = self.f_norm_ref(to_3d(ref_f))

        out_f_tok = self.f_mamba(tar_f_tok, extra_emb=ref_f_tok)
        out_f_tok = out_f_tok + to_3d(tar_f)

        out_f = to_4d(out_f_tok, H, W)
        out_f2 = self.f_out(out_f)              # [B,2C,H,W]
        fr, fi = torch.chunk(out_f2, 2, dim=1)
        Out_F = torch.complex(fr, fi)

        if self.use_fftshift:
            Out_F = ifftshift2(Out_F)

        out_ifft = ifft2(Out_F).real         

        # ========= fuse =========
        out = self.fuse(out_s, out_ifft)
        return out


# ==================== 主网络 Net====================

class Net(nn.Module):

    def __init__(self, num_channels=None, base_filter=None, args=None):
        super().__init__()

        if args is not None:
            self.base_filter  = getattr(args, 'base_filter', 32)
            self.wavelet_name = getattr(args, 'wavelet', 'haar')
            self.use_fftshift = getattr(args, 'use_fftshift', False)
        else:
            self.base_filter  = base_filter if base_filter is not None else 32
            self.wavelet_name = 'haar'
            self.use_fftshift = False

        C = self.base_filter

        # ========== Stage1: shallow CNN ==========
        self.t1_encoder = nn.Sequential(
            nn.Conv2d(1, C, 3, 1, 1),
            HinResBlock(C, C),HinResBlock(C, C),HinResBlock(C, C),
        )
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(1, C, 3, 1, 1),
            HinResBlock(C, C),HinResBlock(C, C),HinResBlock(C, C),
        )

        # ========== Stage2: SingleMamba (long-range) ==========
        self.t1_single = nn.Sequential(
            *[MambaBlock(C) for _ in range(2)] 
        )

        self.t2_single = nn.Sequential(
            *[MambaBlock(C) for _ in range(2)]
        )

        self.dual_pre_dwt = DualDomainCrossMamba(C, use_fftshift=self.use_fftshift)

        wavelet_obj = _as_wavelet(self.wavelet_name)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(wavelet_obj, flip=True)

        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.dwt_l0  = DWT(self.dec_lo, self.dec_hi, wavelet=self.wavelet_name, level=1)
        self.dwt_l1  = DWT(self.dec_lo, self.dec_hi, wavelet=self.wavelet_name, level=1)
        self.idwt_l1 = IDWT(self.rec_lo, self.rec_hi, wavelet=self.wavelet_name, level=1)
        self.idwt_l0 = IDWT(self.rec_lo, self.rec_hi, wavelet=self.wavelet_name, level=1)

        self.fuse = LearnableFusion(C)

        # ========== HF-guided LF blocks ==========
        self.enc_l0_hf = nn.ModuleList([HighFreqGuidedMamba(C) for _ in range(1)])
        self.enc_l1_hf = nn.ModuleList([HighFreqGuidedMamba(C) for _ in range(1)])
        self.dec_l1_hf = nn.ModuleList([HighFreqGuidedMamba(C) for _ in range(1)])
        self.dec_l0_hf = nn.ModuleList([HighFreqGuidedMamba(C) for _ in range(1)])

        # self.skip_l0   = nn.Conv2d(C * 2, C, 3, 1, 1)
        self.skip_l0 = nn.Conv2d(C * 2, C, 1, 1, 0) 
        self.skip_full = nn.Conv2d(C * 2, C, 1, 1, 0)
        self.cross_mamba_full = FRBMamba(C)

        self.dual_post_full = DualDomainCrossMamba(C, use_fftshift=self.use_fftshift)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1),
            # nn.GELU()
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.output = Refine(C, 1)



    def forward(self, t2, t1):
        """
        t2: [B,1,H,W] target
        t1: [B,1,H,W] ref
        """

        B, _, H0, W0 = t2.shape

        # ================= Stage1: shallow =================
        t1_shallow = self.t1_encoder(t1)  # [B,C,H,W]
        t2_shallow = self.t2_encoder(t2)

        # ================= Stage2: SingleMamba  =================
        t1_long = self.t1_single(t1_shallow)  # [B,C,H,W]
        t2_long = self.t2_single(t2_shallow)

        # ================= pre SSDI =================
        # tar 被 ref 条件引导，对齐后再进入 wavelet 金字塔
        t2_align = self.dual_pre_dwt(t2_long, t1_long)  # [B,C,H,W]

        # ================= Encoder Level0 (H/2) =================
        L0_ref, (LH0_ref, HL0_ref, HH0_ref) = self.dwt_l0(t1_long)
        L0_tar, (LH0_tar, HL0_tar, HH0_tar) = self.dwt_l0(t2_align)

        # subband fuse
        L0  = self.fuse(L0_tar,  L0_ref)
        LH0 = self.fuse(LH0_tar, LH0_ref)
        HL0 = self.fuse(HL0_tar, HL0_ref)
        HH0 = self.fuse(HH0_tar, HH0_ref)

        H0_cat = torch.cat([LH0, HL0, HH0], dim=1)  # [B,3C,H/2,W/2]

        for blk in self.enc_l0_hf:
            L0 = blk(L0, H0_cat)
        L0_skip = L0

        # ================= Encoder Level1 (H/4) =================
        L1, (LH1, HL1, HH1) = self.dwt_l1(L0)
        H1_cat = torch.cat([LH1, HL1, HH1], dim=1)  # [B,3C,H/4,W/4]

        for blk in self.enc_l1_hf:
            L1 = blk(L1, H1_cat)

        # ================= Decoder Level1 =================
        for blk in self.dec_l1_hf:
            L1 = blk(L1, H1_cat)

        L0_dec = self.idwt_l1([L1, (LH1, HL1, HH1)], None)          # back to H/2
        L0_dec = self.skip_l0(torch.cat([L0_dec, L0_skip], dim=1))  # skip

        # ================= Decoder Level0 =================
        for blk in self.dec_l0_hf:
            L0_dec = blk(L0_dec, H0_cat)

        # back to full-res
        full_feat = self.idwt_l0([L0_dec, (LH0, HL0, HH0)], None)   # [B,C,H,W]

        # full skip
        full_feat = self.skip_full(torch.cat([full_feat, t2_align], dim=1))

        # full-res cross
        full_feat = self.cross_mamba_full(full_feat, t1_long)
        # ================= post SFDI =================
        full_feat = self.dual_post_full(full_feat, t1_long)

        # ================= final =================
        full_feat = self.final_fusion(full_feat)
        t2_res = self.output(full_feat)
        out = t2_res + t2
        return out



def build_model(args):
    return Net(args=args)


