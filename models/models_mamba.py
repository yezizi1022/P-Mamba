# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights
import math
from pytorch_wavelets import DWTForward
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from .rope import VisionRotaryEmbeddingFast

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            if RMSNorm is None:
                raise ImportError("RMSNorm is not available. Please ensure `mamba_ssm` is installed.")
            if not isinstance(self.norm, (nn.LayerNorm, RMSNorm)):
                raise ValueError("Only LayerNorm and RMSNorm are supported for fused_add_norm.")

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    """
    Creates a single block with a mixer and normalization.

    Args:
        d_model: Dimension of the model.
        ssm_cfg: Configuration for the state-space model.
        norm_epsilon: Epsilon for LayerNorm or RMSNorm.
        drop_path: Probability of dropping paths.
        rms_norm: Whether to use RMSNorm instead of LayerNorm.
        residual_in_fp32: Whether to keep residuals in fp32.
        fused_add_norm: Whether to fuse add and norm operations.
        layer_idx: Layer index for debugging or specific logic.
        device: Device to place the model on.
        dtype: Data type for tensors.
        bimamba_type: Type of BiMamba model to use.

    Returns:
        A Block instance.
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # Mixer class
    mixer_cls = partial(
        Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs
    )

    # Norm class
    if rms_norm:
        if RMSNorm is None:
            raise ImportError("RMSNorm is not available. Please ensure `mamba_ssm` is installed.")
        norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    else:
        norm_cls = partial(nn.LayerNorm, normalized_shape=d_model, eps=norm_epsilon, **factory_kwargs)

    # Instantiate Block
    block = Block(
        dim=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def PMDiffusion(LH, HL, k=1):
    # Detach and move tensors to CPU
    LH_detached = LH.detach().cpu().numpy()
    HL_detached = HL.detach().cpu().numpy()

    grad_u_detached = torch.sqrt(LH ** 2 + HL ** 2).detach().cpu().numpy()

    # Compute the edge-stopping function g(|grad(u)|)
    g_grad_u = 1 / (1 + (grad_u_detached / k) ** 2)

    # Compute the divergence of the diffusion flux
    div_g_grad_u_x = np.gradient(g_grad_u * LH_detached, axis=3)
    div_g_grad_u_y = np.gradient(g_grad_u * HL_detached, axis=2)
    div_g_grad = div_g_grad_u_x + div_g_grad_u_y

    # Convert back to PyTorch tensor
    div_g_grad_tensor = torch.from_numpy(div_g_grad).to(LH.device)

    return div_g_grad_tensor

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class VisionMambaSingle(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.DWTForward = DWTForward(wave='haar')  # 初始化DWTForward

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim)   #224 16 192
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )


        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        # self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.weighting_layer_t1_att1 = nn.Linear(embed_dim, 1)
        self.weighting_layer_s2_att2 = nn.Linear(embed_dim, 1)
        self.BasicResidualBlock = BasicResidualBlock(embed_dim,embed_dim)


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)         #[4, 3, 256, 256] --> [48, 4096, 256] =4096=64*64
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks [1,1,192]-->[4,1,192]
            x = torch.cat((cls_token, x), dim=1)        # [4,1,192] cat [4,196,192] == [4,197,196]
        #走这里
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            # Additional PMD operations
            hidden_states_pmd = x.permute(0, 2, 1).reshape(
                B, self.embed_dim, self.img_size // self.patch_size, self.img_size // self.patch_size
            )
            hidden_states_pmd_size = hidden_states_pmd.shape[2:]

            # Apply DWT and Perona-Malik Diffusion
            LL, yh = self.DWTForward(hidden_states_pmd)
            LH, HL = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :]
            hidden_states_pmd = PMDiffusion(LH, HL, k=1)  # Apply Perona-Malik diffusion
            hidden_states_pmd = F.interpolate(hidden_states_pmd, size=hidden_states_pmd_size, mode='bilinear',
                                              align_corners=True)

            # Apply Basic Residual Block
            hidden_states_pmd = self.BasicResidualBlock(hidden_states_pmd)

            hidden_states_pmd = hidden_states_pmd.view(B, self.embed_dim, -1)
            hidden_states_pmd = hidden_states_pmd.permute(0, 2, 1)

            # Get t1 and att1 weights
            w_t1_att1 = torch.sigmoid(self.weighting_layer_t1_att1(x.mean(dim=1))).unsqueeze(-1)  # Apply sigmoid

            # Update hidden_states with weighted combination
            hidden_states = hidden_states + self.drop_path(w_t1_att1 * hidden_states_pmd + (1 - w_t1_att1) * hidden_states)


        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:#走这里
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.if_cls_token:#True
            return hidden_states[:, 0, :]
        return hidden_states    #torch.Size([batch_size, 14*14, 192])
        

    def forward(self, x, return_features=False, inference_params=None):
        # print(f"x.shape:{x.shape}")                         #x.shape:torch.Size([4, 3, 224, 224])
        x = self.forward_features(x, inference_params)

        # print(f"feature x.shape:{x.shape}")                 #feature x.shape:torch.Size([4, 28*28,embed_dim])
        W = int(math.sqrt(x.shape[1]))

        out0 = x.reshape(-1,W,W,self.embed_dim).permute(0,3,1,2).contiguous()#[32, 64, 28, 28]

        return out0
    
class VisionMambaConcat(nn.Module):
    def __init__(self,img_size):
        super().__init__()
        self.embed_dim = 128
        self.mamba0 = VisionMambaSingle(img_size=img_size,patch_size=4, embed_dim=self.embed_dim,
                                        depth=2, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                        final_pool_type='all', if_abs_pos_embed=True, 
                                        if_rope=True, if_rope_residual=True, bimamba_type="v2")

        self.mamba1 = VisionMambaSingle(img_size=64,patch_size=2, embed_dim=self.embed_dim*2, channels = 128,
                                    depth=2, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                    final_pool_type='all', if_abs_pos_embed=True, 
                                    if_rope=True, if_rope_residual=True, bimamba_type="v2")

        self.mamba2 = VisionMambaSingle(img_size=32, patch_size=2, embed_dim=self.embed_dim* 4, channels = 256,
                                        depth=2, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                        final_pool_type='all', if_abs_pos_embed=True,
                                        if_rope=True, if_rope_residual=True, bimamba_type="v2")

        self.mamba3 = VisionMambaSingle(img_size=16, patch_size=2, embed_dim=self.embed_dim * 8,  channels = 512,
                                        depth=2, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                                        final_pool_type='all', if_abs_pos_embed=True,
                                        if_rope=True, if_rope_residual=True, bimamba_type="v2")


    def init_weights(self, pretrained=None, strict=False):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     load_checkpoint(self, pretrained, strict=strict)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')
        #下面是用来生成featuremap
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(checkpoint, strict=strict)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        out0 = self.mamba0(x)                   # [batch_size,64,28,28]
        out1 = self.mamba1(out0)
        out2 = self.mamba2(out1)
        out3 = self.mamba3(out2)
        output=[out0,out1,out2,out3]

        return output