import copy
import json
import logging
import math
import re
import os, time, json
import types
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import transformers
import av
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_seek_uniform

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from collections import Counter, defaultdict


def _patch_transformers_for_llava():
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
    from transformers.models.qwen2 import modeling_qwen2

    if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
        transformers.modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not getattr(modeling_qwen2.Qwen2RotaryEmbedding, "_densevideo_rope_patch", False):
        original_init = modeling_qwen2.Qwen2RotaryEmbedding.__init__

        def patched_init(self, *args, **kwargs):
            config = kwargs.get("config")
            if config is None and args and not isinstance(args[0], (int, float)):
                config = args[0]
            if config is not None and getattr(config, "rope_parameters", None) is None:
                config.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": getattr(config, "rope_theta", 1000000.0),
                }
            return original_init(self, *args, **kwargs)

        patched_init._densevideo_rope_patch = True
        modeling_qwen2.Qwen2RotaryEmbedding.__init__ = patched_init
        modeling_qwen2.Qwen2RotaryEmbedding._densevideo_rope_patch = True
    try:
        from llava.model.multimodal_encoder import siglip_encoder
    except ImportError:
        siglip_encoder = None
    if siglip_encoder is not None and not hasattr(siglip_encoder.SigLipVisionConfig, "_set_token_in_kwargs"):
        @classmethod
        def _set_token_in_kwargs(cls, kwargs, token=None):
            if token is not None:
                kwargs["token"] = token
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")

        siglip_encoder.SigLipVisionConfig._set_token_in_kwargs = _set_token_in_kwargs


_patch_transformers_for_llava()

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionEmbeddings


HF_HOME   = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
VIDEO_DIR = os.path.join(HF_HOME, "DenseVideo-LPM", "videos")

try:
    from tools.densevideo.codec_frame_types import (
        frame_type_label_from_av,
        frame_type_summary,
        is_key_frame_type,
        is_p_frame_type,
        read_video_frame_types,
        select_frame_type_labels,
    )
except Exception:
    frame_type_label_from_av = None
    frame_type_summary = None
    is_key_frame_type = None
    is_p_frame_type = None
    read_video_frame_types = None
    select_frame_type_labels = None

def get_video_path(old_path: str) -> str:
    if HF_HOME in old_path:
        return old_path
    new_path = os.path.join(VIDEO_DIR, os.path.basename(old_path))
    if not os.path.isfile(new_path):
        raise FileNotFoundError(f"视频不存在: {new_path}")
    return new_path


def resolve_video_path(path_like: str) -> str:
    """
    Resolve dataset-provided relative video paths to local cache locations.
    This makes runs robust when parquet stores legacy prefixes like
    "DenseVideo-LPM/videos/*.mp4" but local cache uses HF_HOME subfolders.
    """
    if os.path.isfile(path_like):
        return path_like

    base = os.path.basename(path_like)
    candidates = [
        os.path.join(os.getcwd(), path_like),
        os.path.join(HF_HOME, path_like),
        os.path.join(HF_HOME, "DenseVideoEvaluation", path_like),
        os.path.join(HF_HOME, "highmotion_densevideounderstand", path_like),
        os.path.join(HF_HOME, "highmotion_densevideounderstand", "all_test", path_like),
        os.path.join(HF_HOME, "DenseVideoEvaluation", "videos", base),
        os.path.join(HF_HOME, "highmotion_densevideounderstand", "videos", base),
        os.path.join(HF_HOME, "DenseVideo-LPM", "videos", base),
        os.path.join(os.getcwd(), "DenseVideo-LPM", "videos", base),
        os.path.join(os.getcwd(), ".cache", "hf", "DenseVideoEvaluation", "videos", base),
        os.path.join(os.getcwd(), ".cache", "hf", "highmotion_densevideounderstand", "videos", base),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return path_like




class GatedVisionEmbeddings(nn.Module):
    """
    Drop-in replacement for SigLipVisionEmbeddings that
    (a) only re-computes moving patches via GatedVideoPatchEmbed logic,
    (b) scatters them back into a full [B*F, N, D] sequence,
    (c) adds *one* positional embedding, and
    (d) returns exactly the [B*Fr, N, D] tensor the ViT encoder wants.
    When `parent.profiling=True`, it measures the gating time with CUDA events.
    """
    def __init__(self,
                 orig_embeds: SigLipVisionEmbeddings,
                 diff_threshold: float, # low to presevre,  high to drop
                 gate_policy: str = "motion",
                 gate_metric: str = "l2",
                 random_keep_ratio: Optional[float] = None,
                 random_seed: int = 0,
                 matched_keep_ratio: Optional[float] = None,
                 gate_seed: Optional[int] = None,
                 parent=None):
        super().__init__()
        self.parent = parent

        # patch params
        self.patch_size  = orig_embeds.patch_size
        self.num_patches = orig_embeds.num_patches  # N
        self.embed_dim   = orig_embeds.embed_dim    # D

        # flatten conv→linear weights
        orig_conv = orig_embeds.patch_embedding
        D, C, p, _ = orig_conv.weight.shape
        self.orig_conv = orig_conv
        self.weight_flat = orig_conv.weight.view(D, -1)  # [D, C*p*p]
        self.bias        = orig_conv.bias                # [D]

        # positional embedding
        self.pos_embed = orig_embeds.position_embedding  # Embedding(N, D)

        # gating threshold
        self.diff_threshold = diff_threshold
        self.gate_policy = str(gate_policy or "motion").lower()
        if self.gate_policy not in {"motion", "random", "uniform", "all", "codec"}:
            raise ValueError(f"Unsupported gate_policy={gate_policy}. Expected motion|random|uniform|all|codec.")
        self.gate_metric = str(gate_metric or "l2").lower()
        if self.gate_metric not in {"l2", "ssim"}:
            raise ValueError(f"Unsupported gate_metric={gate_metric}. Expected l2|ssim.")
        self.random_keep_ratio = self._parse_optional_ratio(random_keep_ratio)
        self.matched_keep_ratio = self._parse_optional_ratio(matched_keep_ratio)
        if self.random_keep_ratio is not None:
            self.random_keep_ratio = min(max(self.random_keep_ratio, 0.0), 1.0)
        if self.matched_keep_ratio is not None:
            self.matched_keep_ratio = min(max(self.matched_keep_ratio, 0.0), 1.0)
        self.random_seed = int(gate_seed if gate_seed is not None else random_seed)
        self._random_call_idx = 0

        # keep for hook + output
        self.register_buffer("position_ids", orig_embeds.position_ids.clone())
        self.last_keep_flat = None
        self.last_diff_score_flat = None

        # prepare CUDA events if profiling
        if parent is not None and getattr(parent, "profiling", False):
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt   = torch.cuda.Event(enable_timing=True)

    @staticmethod
    def _parse_optional_ratio(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "auto"}:
            return None
        return float(value)

    def _codec_keyframe_mask(self, fr: int, device) -> Optional[torch.Tensor]:
        parent = self.parent
        if parent is None:
            return None
        frame_types = getattr(parent, "_last_frame_types", None)
        if not frame_types or len(frame_types) != fr:
            return None
        out = []
        for label in frame_types:
            label = str(label).upper()
            out.append(label.startswith("I") or label == "K" or label == "IDR")
        if out:
            out[0] = True
        return torch.tensor(out, dtype=torch.bool, device=device)

    @staticmethod
    def _patch_ssim_dissimilarity(curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        x = curr.float()
        y = prev.float()
        mu_x = x.mean(dim=-1)
        mu_y = y.mean(dim=-1)
        xc = x - mu_x.unsqueeze(-1)
        yc = y - mu_y.unsqueeze(-1)
        var_x = (xc * xc).mean(dim=-1)
        var_y = (yc * yc).mean(dim=-1)
        cov_xy = (xc * yc).mean(dim=-1)
        dynamic_range = torch.maximum(
            x.amax(dim=-1) - x.amin(dim=-1),
            y.amax(dim=-1) - y.amin(dim=-1),
        ).clamp_min(1.0)
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        ssim = numerator / denominator.clamp_min(1e-12)
        return (1.0 - ssim).clamp(min=0.0, max=2.0)

    # def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    #     parent = self.parent
    #     # --- profiling start ---
    #     if parent is not None and getattr(parent, "profiling", False):
    #         torch.cuda.synchronize()
    #         self.start_evt.record()
    #         # compute total original patches for stats
    #         if pixel_values.dim() == 4:
    #             Bf = pixel_values.shape[0]
    #         else:
    #             Bf = pixel_values.shape[1]
    #         orig_patches = Bf * self.num_patches

    #     # unify shape → [B, Fr, 3, H, W]
    #     if pixel_values.dim() == 4:
    #         BF, C, H, W = pixel_values.shape
    #         pixel_values = pixel_values.unsqueeze(0)
    #     B, Fr, C, H, W = pixel_values.shape

    #     p  = self.patch_size
    #     Hf, Wf = H//p, W//p
    #     N  = Hf * Wf
    #     P  = C * p * p

    #     # extract patches
    #     x_flat  = pixel_values.view(B*Fr, C, H, W)
    #     patches = (x_flat
    #                .unfold(2, p, p)
    #                .unfold(3, p, p)
    #                .permute(0, 2, 4, 1, 3, 5)
    #                .reshape(B, Fr, N, P))

    #     # compute which patches to keep
    #     diffs = (patches[:,1:] - patches[:,:-1]).norm(dim=-1)  # [B, Fr-1, N]
    #     keep  = torch.zeros(B, Fr, N, device=patches.device, dtype=torch.bool)
    #     keep[:,0,:] = True
    #     keep[:,1:,:] = diffs > self.diff_threshold

    #     # flatten + mask
    #     keep_flat      = keep.reshape(B*Fr, N)           # [B*Fr, N]
    #     self.last_keep_flat = keep_flat
    #     patches_flat   = patches.reshape(B*Fr, N, P)     # [B*Fr, N, P]
    #     mask_flat      = keep_flat.view(-1)             # [B*Fr*N]
    #     kept_vecs      = patches_flat.reshape(-1, P)[mask_flat]  # [M, P]

    #     # linear projection
    #     emb_kept = F.linear(kept_vecs, self.weight_flat, self.bias)  # [M, D]
    #     emb_full = emb_kept.new_zeros(B*Fr, N, self.embed_dim)
    #     emb_full.view(-1, self.embed_dim)[mask_flat] = emb_kept

    #     # add positional
    #     pos_ids = torch.arange(N, device=emb_full.device)
    #     emb_full = emb_full + self.pos_embed(pos_ids).unsqueeze(0)
    #     # emb_full = emb_full[keep_flat]        #TODO: 这里有bug，emb_full的维度不对，后面会报错，需要处理，暂且将keep作为pruning率; 好像是传到后面的merge那里绕过了这个问题，但是会带来额外的cost

    #     ratio = keep_flat.float().mean().item()
    #     print(f"tokenization ratio is {ratio}")
    #     return emb_full

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        # # ----- fast path: 完全关闭 gating，走原始 Conv2d -----
        # if self.diff_threshold <= 0:
        #     # 统一成 [BF, C, H, W]
        #     if pixel_values.dim() == 5:
        #         B, Fr, C, H, W = pixel_values.shape
        #         x = pixel_values.view(B*Fr, C, H, W)
        #     else:
        #         x = pixel_values  # [BF, C, H, W]

        #     # 原生 patch-embed
        #     feats = self.orig_conv(x)                   # [BF, D, Hf, Wf]
        #     feats = feats.flatten(2).transpose(1, 2)    # [BF, N, D]

        #     # 加一次位置编码（与原实现一致）
        #     pos_ids = torch.arange(self.num_patches, device=feats.device)
        #     feats = feats + self.pos_embed(pos_ids).unsqueeze(0)

        #     # 给下游 hook 用：全 True
        #     self.last_keep_flat = torch.ones(
        #         feats.shape[0], self.num_patches, dtype=torch.bool, device=feats.device
        #     )
        #     return feats

        parent = self.parent

        # ---------- profiling ----------
        if parent is not None and getattr(parent, "profiling", False):
            torch.cuda.synchronize()
            self.start_evt.record()

        # 统一成 [B, Fr, C, H, W] 与 [BF, C, H, W]
        if pixel_values.dim() == 4:                           # [BF, C, H, W]
            BF, C, H, W = pixel_values.shape
            B, Fr = 1, BF
            x_bchw = pixel_values.contiguous()
        else:                                                 # [B, Fr, C, H, W]
            B, Fr, C, H, W = pixel_values.shape
            x_bchw = pixel_values.view(B*Fr, C, H, W).contiguous()

        p  = self.patch_size
        Hf, Wf = H // p, W // p
        N  = Hf * Wf
        P  = C * p * p
        BF = B * Fr

        # ---- 用 F.unfold 精确对齐 Conv2d 的 im2col 顺序 ----
        patches = torch.nn.functional.unfold(x_bchw, kernel_size=p, stride=p)   # [BF, C*p*p, N]
        patches = patches.transpose(1, 2).contiguous()                           # [BF, N, P]
        # 与权重 dtype 对齐（防止 fp16/fp32 混用）
        if patches.dtype != self.weight_flat.dtype:
            patches = patches.to(self.weight_flat.dtype)
        patches = patches.view(B, Fr, N, P)                                      # [B, Fr, N, P]

        keep = torch.zeros(B, Fr, N, device=patches.device, dtype=torch.bool)
        score = torch.zeros(B, Fr, N, device=patches.device, dtype=torch.float32)
        keep[:, 0, :]  = True
        score[:, 0, :] = 1.0
        if Fr > 1:
            if self.gate_metric == "ssim":
                diffs = self._patch_ssim_dissimilarity(patches[:, 1:], patches[:, :-1])  # [B, Fr-1, N]
            else:
                diffs = (patches[:, 1:] - patches[:, :-1]).norm(dim=-1)              # [B, Fr-1, N]
            motion_keep = diffs > self.diff_threshold
            diffs_f = diffs.float()
            denom = diffs_f.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            motion_score = diffs_f / denom
            if self.diff_threshold < 0:
                motion_keep.fill_(True)
                motion_score.fill_(1.0)
            if self.gate_policy == "all":
                keep[:, 1:, :].fill_(True)
                score[:, 1:, :].fill_(1.0)
            elif self.gate_policy == "motion":
                keep[:, 1:, :] = motion_keep
                score[:, 1:, :] = motion_score
            elif self.gate_policy == "codec":
                keep[:, 1:, :] = motion_keep
                score[:, 1:, :] = motion_score
                key_mask = self._codec_keyframe_mask(Fr, patches.device)
                if key_mask is not None:
                    keep[:, key_mask, :].fill_(True)
                    score[:, key_mask, :].fill_(1.0)
            else:
                if self.matched_keep_ratio is not None:
                    target_counts = torch.full((B, Fr - 1), int(round(N * self.matched_keep_ratio)), dtype=torch.long, device=patches.device)
                elif self.random_keep_ratio is not None:
                    target_counts = torch.full((B, Fr - 1), int(round(N * self.random_keep_ratio)), dtype=torch.long, device=patches.device)
                else:
                    target_counts = motion_keep.sum(dim=-1).long()
                target_counts = target_counts.clamp(min=0, max=N)
                if self.gate_policy == "random":
                    gen = torch.Generator(device=patches.device)
                    gen.manual_seed(self.random_seed + self._random_call_idx)
                    self._random_call_idx += 1
                    rand = torch.rand((B, Fr - 1, N), device=patches.device, generator=gen)
                    for b in range(B):
                        for t in range(Fr - 1):
                            k = int(target_counts[b, t].item())
                            if k > 0:
                                idx = torch.topk(rand[b, t], k=k, largest=False, sorted=False).indices
                                keep[b, t + 1, idx] = True
                    score[:, 1:, :] = keep[:, 1:, :].float()
                elif self.gate_policy == "uniform":
                    for b in range(B):
                        for t in range(Fr - 1):
                            k = int(target_counts[b, t].item())
                            if k > 0:
                                idx = torch.linspace(0, N - 1, steps=k, device=patches.device).round().long().unique()
                                if idx.numel() < k:
                                    pad = torch.arange(N, device=patches.device)[: k - idx.numel()]
                                    idx = torch.cat([idx, pad]).unique()[:k]
                                keep[b, t + 1, idx[:k]] = True
                    score[:, 1:, :] = keep[:, 1:, :].float()
        self.last_keep_flat = keep.view(BF, N)                                    # 给下游 hook
        self.last_diff_score_flat = score.view(BF, N)

        # ---- 第 0 帧全量线性映射（等价 Conv2d(kernel=p, stride=p)）----
        p0 = patches[:, 0].reshape(-1, N, P)                                      # [B, N, P]
        e0 = torch.nn.functional.linear(p0.reshape(-1, P), self.weight_flat, self.bias) \
                .view(-1, N, self.embed_dim)                                      # [B, N, D]

        emb_full = e0.new_empty(BF, N, self.embed_dim)                            # [BF, N, D]
        emb_full[:B] = e0

        # ---- 后续帧：只对 keep==True 重算，其余复用上一帧 ----
        for t in range(1, Fr):
            prev  = emb_full[(t-1)*B : t*B].clone()                                # [B, N, D]
            kt    = keep[:, t]                                                     # [B, N] bool
            if kt.any():
                pt      = patches[:, t].reshape(-1, N, P)                          # [B, N, P]
                kt_flat = kt.reshape(-1)                                           # [B*N]
                et_sel  = torch.nn.functional.linear(
                    pt.reshape(-1, P)[kt_flat], self.weight_flat, self.bias
                )                                                                  # [M, D]
                prev.view(-1, self.embed_dim)[kt_flat] = et_sel
            emb_full[t*B : (t+1)*B] = prev

        # ---- 位置编码：严格用原模块的 position_ids 对齐网格 ----
        pos_ids = self.position_ids[:N].to(device=emb_full.device)

        # ---- 位置编码：严格用原模块的 position_ids 对齐网格 ----
        # 让 pos_ids 一定是一维 [N]
        if self.position_ids.dim() == 2:
            pos_ids = self.position_ids[0, :N]
        else:
            pos_ids = self.position_ids[:N]

        pos = self.pos_embed(pos_ids.to(device=emb_full.device))     # [N, D]
        if pos.dtype != emb_full.dtype:
            pos = pos.to(emb_full.dtype)

        # 直接相加（[BF, N, D] + [N, D] 会在 batch 维自动广播）
        emb_full = emb_full + pos

        # Always record keep ratio for downstream tradeoff parsing.
        if parent is not None:
            orig_patches = (B * Fr) * N
            recomputed = int(keep.sum().item())
            ratio_keep = recomputed / float(orig_patches) if orig_patches else 1.0
            parent._last_gate_keep_ratio = float(ratio_keep)
            parent._last_recomputed_patches = int(recomputed)
            parent._last_orig_patches = int(orig_patches)
            parent._last_recompute_ratio = float(ratio_keep)
            parent._last_gate_policy = self.gate_policy
            parent._last_gate_metric = self.gate_metric


        # ---------- profiling ----------
        if parent is not None and getattr(parent, "profiling", False):
            torch.cuda.synchronize()
            self.end_evt.record()
            torch.cuda.synchronize()
            print(
                f"[Gated] policy={self.gate_policy} recompute ratio (patch-embed) = "
                f"{ratio_keep:.3f} ({recomputed}/{orig_patches})"
            )

        return emb_full




@register_model("llava_onevision_dense")
class Llava_OneVision_Dense(lmms):
    """

    self.model
        LlavaQwenForCausalLM(
        (model): LlavaQwenModel(
            (embed_tokens): Embedding(151647, 896)
            (layers): ModuleList(
            (0-23): 24 x Qwen2DecoderLayer(
                (self_attn): Qwen2SdpaAttention(
                (q_proj): Linear(in_features=896, out_features=896, bias=True)
                (k_proj): Linear(in_features=896, out_features=128, bias=True)
                (v_proj): Linear(in_features=896, out_features=128, bias=True)
                (o_proj): Linear(in_features=896, out_features=896, bias=False)
                (rotary_emb): Qwen2RotaryEmbedding()
                )
                (mlp): Qwen2MLP(
                (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
                (up_proj): Linear(in_features=896, out_features=4864, bias=False)
                (down_proj): Linear(in_features=4864, out_features=896, bias=False)
                (act_fn): SiLU()
                )
                (input_layernorm): Qwen2RMSNorm()
                (post_attention_layernorm): Qwen2RMSNorm()
            )
            )
            (norm): Qwen2RMSNorm()
            (vision_tower): SigLipVisionTower(
            (vision_tower): SigLipVisionModel(
                (vision_model): SigLipVisionTransformer(
                (embeddings): SigLipVisionEmbeddings(
                    (patch_embedding): GatedVideoPatchEmbed(
                    (orig_conv): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
                    (pos_embed): Embedding(729, 1152)
                    )
                    (position_embedding): Embedding(729, 1152)
                )
                (encoder): SigLipEncoder(
                    (layers): ModuleList(
                    (0-25): 26 x SigLipEncoderLayer(
                        (self_attn): SigLipAttention(
                        (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                        (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                        (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                        (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
                        )
                        (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                        (mlp): SigLipMLP(
                        (activation_fn): PytorchGELUTanh()
                        (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                        (fc2): Linear(in_features=4304, out_features=1152, bias=True)
                        )
                        (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                    )
                    )
                )
                (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                (head): Identity()
                )
            )
            )
            (vision_resampler): IdentityMap()
            (mm_projector): Sequential(
            (0): Linear(in_features=1152, out_features=896, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=896, out_features=896, bias=True)
            )
        )
        (lm_head): Linear(in_features=896, out_features=151647, bias=False)
        )

    Llava Model
    ┌───────────────────────────────────────────────────────────────────┐
    │                          Input Stage                              │
    │                                                                   │
    │  Video Frames ──▶ [Vision Tower: patch_embedding → ViT encoder → post_LN ]
    │                                   │
    │                                   ▼
    │                          SigLip post_LN (✱)
    │                                   │
    │                              mm_projector
    │                                   │
    │                           Vision Embeddings
    │                                   │
    └───────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                          Text Branch                              │
    │                                                                   │
    │  Raw Question   ──▶ Tokenizer ──▶ Text Token IDs ──▶ embed_tokens │
    │                                   │
    │                                   ▼
    │                             Text Embeddings
    └───────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                          Fusion & Decoder                         │
    │                                                                   │
    │ 1. 在 Prompt 中，我们用特殊的〈image〉标记占位：                       │
    │                                                                   │
    │     ["What", "is", "happening", 〈image〉, "now", "?"]            │
    │                                                                   │
    │ 2. LLaVA 在内部把这些〈image〉Token 对应的位置，                      │
    │    替换成上一步得到的 Vision Embeddings。                         │
    │                                                                   │
    │      ┌────────────┐       ┌────────────────────┐                   │
    │      │ Text Embeds│ ←───▶ │ Multi-modal Embeds │──▶ Qwen2Decoder   │
    │      └────────────┘       └────────────────────┘                   │
    │              ▲                      ▲                               │
    │              │                      │                               │
    │      embed_tokens             mm_projector                         │
    │                                                                   │
    │ 3. Qwen2DecoderLayer（24 层 Transformer）对这串混合了文字和视觉 │
    │    的 embedding 进行自注意力（self-attention），                           │
    │    Text ↔ Vision 完全交互。                                          │
    │                                                                   │
    │ 4. 最后经过 LM head → Softmax 预测下一个 token。                  │
    └───────────────────────────────────────────────────────────────────┘







    modified dense video understanding:
        raw pixels ─▶ gated patch embed ─▶ ViT encoder ─▶ post_LN ─▶ mm_projector ─▶┐
                                                                            │  <-- hook 在这里做 scene-aware merge
                                        merged vision_feats ────────────────────┘
                                                                            │
    text_tokens ─▶ embed_tokens ────────────────────────────────────────────────▶│
                                                                            ▼
                                            Qwen2Decoder( text + vision ) …


    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        dense_frame_fps: Optional[float] = 2.0,
        use_gated_tok: Optional[bool] = True,
        use_vision_merge: Optional[bool] = True,
        gate_diff_threshold: Optional[float] = 0.3,
        gate_policy: Optional[str] = "motion",
        gate_metric: Optional[str] = "l2",
        use_codec_frame_types: Optional[bool] = False,
        random_keep_ratio: Optional[float] = None,
        random_seed: Optional[int] = 0,
        matched_keep_ratio: Optional[float] = None,
        gate_seed: Optional[int] = None,
        enable_visual_token_pruning: Optional[bool] = False,
        prune_mode: Optional[str] = "off",
        prune_apply_mode: Optional[str] = "pack",
        prune_keep_ratio: Optional[float] = 1.0,
        prune_min_tokens_per_frame: Optional[int] = 16,
        prune_seed: Optional[int] = 0,
        merge_jsd_threshold: Optional[float] = 0.4,
        scene_cut_factor: Optional[float] = 0.8,
        scene_merge_apply: Optional[bool] = True,
        scene_merge_keep_ratio: Optional[float] = 0.5,
        scene_merge_min_tokens_per_frame: Optional[int] = 16,
        scene_merge_require_semantic_match: Optional[bool] = False,
        frame_sampling_strategy: Optional[str] = "uniform",
        frame_stride: Optional[int] = 1,
        clip_duration_sec: Optional[float] = None,
        force_include_last_frame: Optional[bool] = True,
        high_fps_guard: Optional[bool] = True,
        high_fps_threshold_fps: Optional[float] = 30.0,
        high_fps_max_frames: Optional[int] = 96,
        visual_token_cap: Optional[int] = 65536,
        oom_retry_times: Optional[int] = 2,
        oom_frame_reduce_ratio: Optional[float] = 0.5,
        oom_token_reduce_ratio: Optional[float] = 0.5,
        min_frame_cap: Optional[int] = 8,
        min_token_cap: Optional[int] = 4096,
        profiling: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.profiling = profiling
        # --- profiling containers ---
        # 如果 profiling=False，就不记录
        if profiling:
            self.stats = defaultdict(float)
            # frames, samples counts
            self.stats["samples"] = 0
            self.stats["frames_processed"] = 0
            self.stats["orig_patches"] = 0
            self.stats["kept_patches"] = 0
            self.stats["vision_time"] = 0.0

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num

        self.dense_frame_fps = dense_frame_fps
        self.use_gated_tok = self._parse_bool(use_gated_tok)
        self.use_vision_merge = self._parse_bool(use_vision_merge)
        self.gate_diff_threshold = gate_diff_threshold
        self.gate_policy = str(gate_policy or "motion").lower()
        if self.gate_policy not in {"motion", "random", "uniform", "all", "codec"}:
            raise ValueError(f"Unsupported gate_policy={gate_policy}. Expected motion|random|uniform|all|codec.")
        self.gate_metric = str(gate_metric or "l2").lower()
        if self.gate_metric not in {"l2", "ssim"}:
            raise ValueError(f"Unsupported gate_metric={gate_metric}. Expected l2|ssim.")
        self.use_codec_frame_types = self._parse_bool(use_codec_frame_types) or self.gate_policy == "codec"
        self.random_keep_ratio = self._parse_optional_ratio(random_keep_ratio)
        self.matched_keep_ratio = self._parse_optional_ratio(matched_keep_ratio)
        self.random_seed = int(gate_seed if gate_seed is not None else (random_seed or 0))
        self.enable_visual_token_pruning = self._parse_bool(enable_visual_token_pruning)
        self.prune_mode = str(prune_mode or "off").lower()
        self.prune_apply_mode = str(prune_apply_mode or "pack").lower()
        self.prune_keep_ratio = float(prune_keep_ratio)
        self.prune_min_tokens_per_frame = int(prune_min_tokens_per_frame)
        self.prune_seed = int(prune_seed or 0)
        if self.prune_mode not in {"off", "motion", "random", "uniform", "codec_residual"}:
            raise ValueError(f"Unsupported prune_mode={prune_mode}. Expected off|motion|random|uniform|codec_residual.")
        if self.prune_apply_mode not in {"pack", "mask", "ragged_pack"}:
            raise ValueError(f"Unsupported prune_apply_mode={prune_apply_mode}. Expected pack|mask|ragged_pack.")
        self.merge_jsd_threshold = merge_jsd_threshold
        self.scene_cut_factor = scene_cut_factor
        self.scene_merge_apply = self._parse_bool(scene_merge_apply)
        self.scene_merge_keep_ratio = self._parse_optional_ratio(scene_merge_keep_ratio)
        if self.scene_merge_keep_ratio is not None:
            self.scene_merge_keep_ratio = min(max(float(self.scene_merge_keep_ratio), 0.0), 1.0)
        self.scene_merge_min_tokens_per_frame = int(scene_merge_min_tokens_per_frame)
        self.scene_merge_require_semantic_match = self._parse_bool(scene_merge_require_semantic_match)
        self.frame_sampling_strategy = frame_sampling_strategy
        self.frame_stride = max(int(frame_stride), 1)
        self.clip_duration_sec = clip_duration_sec
        self.force_include_last_frame = force_include_last_frame
        if isinstance(high_fps_guard, str):
            self.high_fps_guard = high_fps_guard.lower() in {"1", "true", "yes", "y"}
        else:
            self.high_fps_guard = bool(high_fps_guard)
        self.high_fps_threshold_fps = float(high_fps_threshold_fps) if high_fps_threshold_fps is not None else 30.0
        self.high_fps_max_frames = int(high_fps_max_frames) if high_fps_max_frames is not None else 96
        self.visual_token_cap = int(visual_token_cap) if visual_token_cap is not None else 65536
        self.oom_retry_times = max(int(oom_retry_times), 0)
        self.oom_frame_reduce_ratio = float(oom_frame_reduce_ratio) if oom_frame_reduce_ratio is not None else 0.5
        self.oom_token_reduce_ratio = float(oom_token_reduce_ratio) if oom_token_reduce_ratio is not None else 0.5
        self.min_frame_cap = max(int(min_frame_cap), 1)
        self.min_token_cap = max(int(min_token_cap), 1)

        # Runtime guard states. These can be reduced on OOM and persist during the run.
        self._dynamic_max_frames_num = int(max_frames_num) if max_frames_num is not None else None
        self._dynamic_visual_token_cap = int(self.visual_token_cap) if self.visual_token_cap and self.visual_token_cap > 0 else None

        # Per-sample metrics for parsing downstream.
        self._last_pre_tokens = None
        self._last_post_tokens = None
        self._last_tokenization_time = None
        self._last_gate_keep_ratio = None
        self._last_recomputed_patches = None
        self._last_orig_patches = None
        self._last_recompute_ratio = None
        self._last_gate_policy = "disabled"
        self._last_gate_metric = self.gate_metric
        self._last_merge_ratio = None
        self._last_sampled_frames = None
        self._last_effective_fps = None
        self._last_video_duration = None
        self._last_pruning_enabled = False
        self._last_prune_mode = "off"
        self._last_prune_apply_mode = self.prune_apply_mode
        self._last_post_tokens_before_prune = None
        self._last_post_tokens_after_prune = None
        self._last_prune_keep_ratio_actual = None
        self._last_ragged_token_mask = None
        self._last_ragged_pooled_tokens = None
        self._last_frame_types = None
        self._last_frame_indices = None
        self._last_frame_type_counts = {}
        self._last_frame_type_source = "none"
        self._last_scene_merge_applied = False
        self._last_scene_merge_before_tokens = None
        self._last_scene_merge_after_tokens = None
        self._last_scene_merge_keep_ratio_actual = 1.0
        self._last_scene_merge_has_semantic_match = False
        self._last_scene_keyframe_mask = None
        self._last_scene_kept_keyframe_mask = None

        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)
        overwrite_config["delay_load"] = True
        mm_tunable_parts = getattr(cfg_pretrained, "mm_tunable_parts", None)
        if isinstance(mm_tunable_parts, str) and "mm_vision_tower" in mm_tunable_parts:
            overwrite_config["mm_tunable_parts"] = ",".join(
                part for part in mm_tunable_parts.split(",") if part != "mm_vision_tower"
            )
        if getattr(cfg_pretrained, "rope_parameters", None) is None:
            overwrite_config["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": getattr(cfg_pretrained, "rope_theta", 1000000.0),
            }

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._patch_packed_prune_pooling()
        self._patch_ragged_video_prepare()



        # ---- 准备累加器 ----
        self._vis_tok_start   = None
        self._vis_tok_time    = 0.0
        self._vis_tok_calls   = 0

        # ---- 找到两个要挂钩的模块 ----
        # 原始的 patch-embed（pixel → tokens）：
        vision_model = self._model.model.vision_tower.vision_tower.vision_model
        emb_module   = vision_model.embeddings

        # mm_projector：把 tokens 投影到 LLM 维度
        proj_module  = self._model.model.mm_projector

        # ---- 注册“前置钩子”：在 pixel 进入 embeddings 前打表 ----
        def _pre_emb(module, inp):
            # 记录开始时间
            # inp[0] 是 pixel_tensor，shape=[B, C, H, W]
            x = inp[0]
            B, C, H, W = x.shape

            # 1) 计算 patch 数量
            p = module.patch_size           # 或者你自己存到 module 上的 patch_size
            Hf = H // p
            Wf = W // p
            N  = Hf * Wf                    # patch token 数量
            P  = C * p * p                  # 每个 patch 的像素数
            eval_logger.info(f"[VisTok] patches: {B}*{Hf}×{Wf}={B*N}, ")
            print(f"[VisTok] patches: {B}*{Hf}×{Wf}={B*N},")

            self._last_pre_tokens = int(B * N)
            self._vis_tok_start = time.perf_counter()

        emb_module.register_forward_pre_hook(_pre_emb)



        # ---- 注册“后置钩子”：在 mm_projector 完成后算差值 ----
        def _post_proj(module, inp, out):
            # 计算 elapsed
            elapsed = time.perf_counter() - self._vis_tok_start
            self._vis_tok_time  += elapsed
            self._vis_tok_calls += 1

            # out.shape == [B, N, hidden]，N 就是 tokenization 后的视觉 token 数量
            batch, num_tokens, dim = out.shape
            self._last_post_tokens = int(batch * num_tokens)
            self._last_tokenization_time = float(elapsed)
            eval_logger.info(
                f"[VisTok] Batches={batch}, After Tokenization Tokens={batch*num_tokens}, "
                f"time={elapsed:.4f}s"
            )
            print(
                f"[VisTok] Batches={batch}, After Tokenization Tokens={batch*num_tokens}, "
                f"total tokenization time={elapsed:.4f}s"
            )

        proj_module.register_forward_hook(_post_proj)



        if self.use_gated_tok:
            # -------------------------------
            # set up hook for GatedVisionEmbeddings:
            # -------------------------------

            # locate original modules
            vision_model = self._model.model.vision_tower.vision_tower.vision_model

            # grab the old SigLip embed block
            old = vision_model.embeddings

            # replace it wholesale
            vision_model.embeddings = GatedVisionEmbeddings(
                orig_embeds   = old,
                diff_threshold=self.gate_diff_threshold,
                gate_policy=self.gate_policy,
                gate_metric=self.gate_metric,
                random_keep_ratio=self.random_keep_ratio,
                random_seed=self.random_seed,
                matched_keep_ratio=self.matched_keep_ratio,
                parent=self,
            )
            new_emb = vision_model.embeddings
            new_emb.register_forward_pre_hook(_pre_emb)


        if self.use_vision_merge or self.enable_visual_token_pruning:

            # --- register a forward‐hook to inject our scene‐aware merge after gating ---
            # gated_embed.register_forward_hook(self._forward_hook)
            # 2) 找到 mm_projector
            mm_proj = self._model.model.mm_projector  # Sequential(Linear→GELU→Linear)

            # 3) 在它上面注册 forward_hook    
            mm_proj.register_forward_hook(self._vision_merge_hook)

            self.jsd_threshold     = self.merge_jsd_threshold
            self.merge_ratio       = 0.5


        # store some config
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num


        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def _parse_bool(self, x):
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return bool(x)
        return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _parse_optional_ratio(self, x):
        if x is None:
            return None
        if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "auto"}:
            return None
        return float(x)

    def _align_prune_aux_tensor(self, tensor, bf: int, n: int, device, dtype=None):
        if tensor is None or not torch.is_tensor(tensor):
            return None
        aux = tensor.detach().to(device=device)
        if aux.dim() != 2:
            if aux.numel() == bf * n:
                aux = aux.reshape(bf, n)
            else:
                return None
        if aux.shape[0] != bf:
            return None
        if aux.shape[1] < n:
            return None
        aux = aux[:, :n]
        if dtype is not None:
            aux = aux.to(dtype=dtype)
        return aux

    def _print_prune_metrics(self):
        before = self._last_post_tokens_before_prune
        after = self._last_post_tokens_after_prune
        ratio = self._last_prune_keep_ratio_actual
        print(
            "[PRUNE_METRICS] "
            f"pruning_enabled={str(bool(self._last_pruning_enabled)).lower()} "
            f"prune_mode={self._last_prune_mode} "
            f"prune_apply_mode={self._last_prune_apply_mode} "
            f"orig_post_tokens={before if before is not None else 0} "
            f"pruned_post_tokens={after if after is not None else 0} "
            f"prune_keep_ratio_actual={(ratio if ratio is not None else 1.0):.6f} "
            f"post_tokens_before_prune={before if before is not None else 0} "
            f"post_tokens_after_prune={after if after is not None else 0}",
            flush=True,
        )

    def _record_prune_metrics(self, enabled: bool, mode: str, apply_mode: str, before: int, after: int):
        self._last_pruning_enabled = bool(enabled)
        self._last_prune_mode = str(mode or "off").lower()
        self._last_prune_apply_mode = str(apply_mode or "pack").lower()
        self._last_post_tokens_before_prune = int(before)
        self._last_post_tokens_after_prune = int(after)
        self._last_prune_keep_ratio_actual = (float(after) / float(before)) if before else 1.0

    def _get_prune_scores(self, out_feats, keep_flat=None):
        bf, n, _ = out_feats.shape
        scores = None
        if self.prune_mode == "motion":
            try:
                gated = self._model.model.vision_tower.vision_tower.vision_model.embeddings
                scores = self._align_prune_aux_tensor(
                    getattr(gated, "last_diff_score_flat", None),
                    bf,
                    n,
                    out_feats.device,
                    dtype=torch.float32,
                )
            except Exception:
                scores = None
            if scores is None:
                scores = self._align_prune_aux_tensor(keep_flat, bf, n, out_feats.device, dtype=torch.float32)
            if scores is None:
                scores = out_feats.float().norm(dim=-1)
        elif self.prune_mode == "random":
            gen = torch.Generator(device=out_feats.device)
            gen.manual_seed(self.prune_seed)
            scores = torch.rand((bf, n), device=out_feats.device, generator=gen, dtype=torch.float32)
        else:
            scores = out_feats.float().norm(dim=-1)
        return scores

    def _get_codec_residual_scores(self, out_feats, keep_flat=None):
        bf, n, _ = out_feats.shape
        try:
            gated = self._model.model.vision_tower.vision_tower.vision_model.embeddings
            residual_scores = self._align_prune_aux_tensor(
                getattr(gated, "last_diff_score_flat", None),
                bf,
                n,
                out_feats.device,
                dtype=torch.float32,
            )
        except Exception:
            residual_scores = None
        if residual_scores is None:
            residual_scores = self._align_prune_aux_tensor(keep_flat, bf, n, out_feats.device, dtype=torch.float32)
        if residual_scores is None:
            residual_scores = out_feats.float().norm(dim=-1)

        denom = residual_scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
        residual_scores = residual_scores / denom

        semantic_scores = out_feats.float().norm(dim=-1)
        semantic_scores = semantic_scores / semantic_scores.amax(dim=1, keepdim=True).clamp_min(1e-12)

        key_mask = self._frame_type_key_mask(bf, out_feats.device, keep_flat=keep_flat)
        residual_frame_mask = self._frame_type_residual_mask(bf, out_feats.device)
        scores = residual_scores.clone()
        if key_mask.any():
            scores[key_mask] = semantic_scores[key_mask] + 1.0
        if (~(key_mask | residual_frame_mask)).any():
            scores[~(key_mask | residual_frame_mask)] = residual_scores[~(key_mask | residual_frame_mask)]
        return scores, key_mask, residual_frame_mask

    def _apply_visual_token_pruning(self, out_feats, keep_flat=None):
        if (
            not self.enable_visual_token_pruning
            or self.prune_mode == "off"
            or not torch.is_tensor(out_feats)
            or out_feats.dim() != 3
        ):
            if torch.is_tensor(out_feats) and out_feats.dim() == 3:
                before = int(out_feats.shape[0] * out_feats.shape[1])
                self._record_prune_metrics(False, "off", self.prune_apply_mode, before, before)
                self._print_prune_metrics()
            return out_feats

        bf, n, d = out_feats.shape
        before = int(bf * n)
        ratio = min(max(float(self.prune_keep_ratio), 0.0), 1.0)
        k = max(int(self.prune_min_tokens_per_frame), int(math.ceil(n * ratio)))
        k = min(max(k, 1), n)

        if self.prune_mode == "codec_residual":
            scores, key_mask, _ = self._get_codec_residual_scores(out_feats, keep_flat=keep_flat)
            if self.prune_apply_mode in {"mask", "ragged_pack"}:
                mask = torch.zeros((bf, n), dtype=torch.bool, device=out_feats.device)
                if key_mask.any():
                    mask[key_mask] = True
                residual_rows = ~key_mask
                if residual_rows.any():
                    idx = torch.topk(scores[residual_rows], k=k, dim=1, largest=True, sorted=False).indices
                    mask[residual_rows] = mask[residual_rows].scatter(1, idx, True)
                if self.prune_apply_mode == "ragged_pack":
                    self._last_ragged_token_mask = mask.detach()
                    pruned = out_feats
                    after = int(mask.sum().item())
                else:
                    pruned = out_feats.masked_fill(~mask.unsqueeze(-1), 0)
                    after = before
            else:
                self._last_ragged_token_mask = None
                idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
                idx, _ = torch.sort(idx, dim=1)
                gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)
                pruned = out_feats.gather(dim=1, index=gather_idx)
                after = int(bf * k)
            self._record_prune_metrics(True, self.prune_mode, self.prune_apply_mode, before, after)
            self._print_prune_metrics()
            return pruned

        if self.prune_mode == "uniform":
            idx_1d = torch.linspace(0, n - 1, steps=k, device=out_feats.device).round().long()
            idx = idx_1d.unsqueeze(0).expand(bf, -1)
        else:
            scores = self._get_prune_scores(out_feats, keep_flat=keep_flat)
            if scores.shape != (bf, n):
                scores = out_feats.float().norm(dim=-1)
            idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
            idx, _ = torch.sort(idx, dim=1)

        if self.prune_apply_mode == "mask":
            mask = torch.zeros((bf, n), dtype=torch.bool, device=out_feats.device)
            mask.scatter_(1, idx, True)
            pruned = out_feats.masked_fill(~mask.unsqueeze(-1), 0)
            after = before
        else:
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)
            pruned = out_feats.gather(dim=1, index=gather_idx)
            after = int(bf * k)

        self._record_prune_metrics(True, self.prune_mode, self.prune_apply_mode, before, after)
        self._print_prune_metrics()
        return pruned

    def _patch_packed_prune_pooling(self):
        needs_prune_patch = (
            self.enable_visual_token_pruning
            and self.prune_mode != "off"
            and self.prune_apply_mode in {"pack", "ragged_pack"}
        )
        needs_scene_merge_patch = self.use_vision_merge and self.scene_merge_apply
        if not (needs_prune_patch or needs_scene_merge_patch):
            return
        if getattr(self._model, "_densevideo_packed_prune_pool_patch", False):
            return

        original_get_2d_pool = self._model.get_2dPool

        def packed_aware_get_2d_pool(model_self, image_feature, stride=2):
            height = width = model_self.get_vision_tower().num_patches_per_side
            if image_feature.dim() != 3 or image_feature.shape[1] == height * width:
                return original_get_2d_pool(image_feature, stride)

            num_frames, num_tokens, _ = image_feature.shape
            pool_stride = max(int(stride), 1)
            if pool_stride <= 1 or num_tokens <= 1:
                return image_feature

            # Packed pruning removes the square patch grid, so the stock 2D
            # pooling view is invalid. Keep the token sequence packed and
            # apply an equivalent stride^2 reduction along the sequence.
            target_tokens = max(int(math.ceil(num_tokens / float(pool_stride * pool_stride))), 1)
            seq = image_feature.transpose(1, 2).contiguous()
            pool_mode = getattr(model_self.config, "mm_spatial_pool_mode", "bilinear")
            if pool_mode == "average":
                seq = F.avg_pool1d(
                    seq,
                    kernel_size=pool_stride * pool_stride,
                    stride=pool_stride * pool_stride,
                    ceil_mode=True,
                )
            elif pool_mode == "max":
                seq = F.max_pool1d(
                    seq,
                    kernel_size=pool_stride * pool_stride,
                    stride=pool_stride * pool_stride,
                    ceil_mode=True,
                )
            elif pool_mode == "bilinear":
                seq = F.interpolate(seq, size=target_tokens, mode="linear", align_corners=False)
            else:
                raise ValueError(f"Unexpected mm_spatial_pool_mode: {pool_mode}")

            if seq.shape[-1] != target_tokens:
                seq = F.interpolate(seq, size=target_tokens, mode="linear", align_corners=False)
            return seq.transpose(1, 2).contiguous()

        self._model._densevideo_original_get_2dPool = original_get_2d_pool
        self._model.get_2dPool = types.MethodType(packed_aware_get_2d_pool, self._model)
        self._model._densevideo_packed_prune_pool_patch = True

    def _apply_ragged_video_pooling(self, model_self, image_feature):
        mask = self._last_ragged_token_mask
        if (
            mask is None
            or not torch.is_tensor(mask)
            or not torch.is_tensor(image_feature)
            or image_feature.dim() != 3
            or mask.shape[0] != image_feature.shape[0]
            or mask.shape[1] != image_feature.shape[1]
        ):
            return model_self.get_2dPool(image_feature).flatten(0, 1)

        mask = mask.to(device=image_feature.device, dtype=torch.bool)
        pooled_frames = []
        for frame_idx in range(image_feature.shape[0]):
            frame_mask = mask[frame_idx]
            selected = image_feature[frame_idx][frame_mask]
            if selected.numel() == 0:
                selected = image_feature[frame_idx][:1]
            pooled = model_self.get_2dPool(selected.unsqueeze(0), getattr(model_self.config, "mm_spatial_pool_stride", 2))
            pooled_frames.append(pooled.reshape(-1, pooled.shape[-1]))
        packed = torch.cat(pooled_frames, dim=0) if pooled_frames else image_feature.new_zeros((0, image_feature.shape[-1]))
        self._last_ragged_pooled_tokens = int(packed.shape[0])
        print(
            "[RAGGED_PACK_METRICS] "
            f"active_tokens_before_pool={int(mask.sum().item())} "
            f"pooled_visual_tokens={self._last_ragged_pooled_tokens}",
            flush=True,
        )
        return packed

    def _patch_ragged_video_prepare(self):
        if not (
            self.enable_visual_token_pruning
            and self.prune_mode == "codec_residual"
            and self.prune_apply_mode == "ragged_pack"
        ):
            return
        if getattr(self._model, "_densevideo_ragged_prepare_patch", False):
            return

        parent = self
        original_prepare = self._model.prepare_inputs_labels_for_multimodal

        def ragged_prepare(
            model_self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            modalities=["image"],
            image_sizes=None,
        ):
            if not (
                parent.enable_visual_token_pruning
                and parent.prune_mode == "codec_residual"
                and parent.prune_apply_mode == "ragged_pack"
            ):
                return original_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

            vision_tower = model_self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
            if not (type(images) is list or getattr(images, "ndim", None) == 5):
                return original_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = [idx for idx, modality in enumerate(modalities) if modality == "video"]
            images_list = []
            for image in images:
                images_list.append(image if image.ndim == 4 else image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = model_self.encode_images(concat_images)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)

            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(parent._apply_ragged_video_pooling(model_self, image_feat))
                else:
                    image_features.append(image_feat)

            mm_patch_merge_type = getattr(model_self.config, "mm_patch_merge_type", "flat")
            if mm_patch_merge_type == "flat":
                image_features = [x if x.dim() == 2 else x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                merged = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.dim() == 2:
                        merged.append(image_feature)
                    elif image_idx in video_idx_in_batch:
                        merged.append(image_feature.flatten(0, 1))
                    else:
                        merged.append(image_feature[0] if image_feature.shape[0] == 1 else image_feature.flatten(0, 1))
                image_features = merged
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {model_self.config.mm_patch_merge_type}")

            if getattr(model_self.config, "tune_mm_mlp_adapter", False) and getattr(model_self.config, "mm_use_im_start_end", False):
                raise NotImplementedError

            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

            new_input_embeds = []
            new_labels = []
            cur_image_idx = 0
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = model_self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    cur_image_idx += 1
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

                text_split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = model_self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, text_split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        try:
                            cur_image_features = image_features[cur_image_idx]
                        except IndexError:
                            cur_image_features = image_features[cur_image_idx - 1]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                device = getattr(model_self, "device", input_ids[0].device)
                cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            tokenizer_model_max_length = getattr(model_self.config, "tokenizer_model_max_length", None)
            new_input_embeds = [x[:tokenizer_model_max_length] for x, _ in zip(new_input_embeds, modalities)]
            new_labels = [x[:tokenizer_model_max_length] for x, _ in zip(new_labels, modalities)]

            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)
            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask_out = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids_out = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(model_self.config, "tokenizer_padding_side", "right") == "left":
                    new_input_embeds_padded.append(
                        torch.cat(
                            (
                                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                                cur_new_embed,
                            ),
                            dim=0,
                        )
                    )
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask_out[i, -cur_len:] = True
                        position_ids_out[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(
                        torch.cat(
                            (
                                cur_new_embed,
                                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                            ),
                            dim=0,
                        )
                    )
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask_out[i, :cur_len] = True
                        position_ids_out[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
            new_labels = None if _labels is None else new_labels_padded
            attention_mask_final = None if _attention_mask is None else attention_mask_out.to(dtype=_attention_mask.dtype)
            position_ids_final = None if _position_ids is None else position_ids_out
            return None, position_ids_final, attention_mask_final, past_key_values, new_input_embeds, new_labels

        self._model._densevideo_original_prepare_inputs_labels_for_multimodal = original_prepare
        self._model.prepare_inputs_labels_for_multimodal = types.MethodType(ragged_prepare, self._model)
        self._model._densevideo_ragged_prepare_patch = True

    def _wants_codec_frame_types(self):
        strategy = str(self.frame_sampling_strategy or "").lower()
        return self.use_codec_frame_types or self.gate_policy == "codec" or self.prune_mode == "codec_residual" or strategy in {
            "codec",
            "codec_ip",
            "codec_kp",
            "codec_keyframe",
            "codec_iframe",
        }

    @staticmethod
    def _is_codec_key_entry(item):
        label = str(item.get("pict_type", "")).upper()
        return bool(item.get("is_keyframe")) or label.startswith("I") or label in {"K", "IDR"}

    @staticmethod
    def _is_codec_residual_entry(item):
        label = str(item.get("pict_type", "")).upper()
        return bool(item.get("is_p_frame")) or label.startswith(("P", "B"))

    @classmethod
    def _select_codec_key_or_residual_indices(cls, entries, desired_count: int, include_residual: bool) -> List[int]:
        key_indices = [int(item["index"]) for item in entries if cls._is_codec_key_entry(item)]
        if not key_indices:
            key_indices = [0]

        if not include_residual:
            frame_indices = sorted(set(key_indices))
            if desired_count > 0 and len(frame_indices) > desired_count:
                select = np.linspace(0, len(frame_indices) - 1, desired_count, dtype=int).tolist()
                frame_indices = [frame_indices[i] for i in select]
            return sorted(set(int(i) for i in frame_indices))

        key_set = set(key_indices)
        residual_indices = [
            int(item["index"])
            for item in entries
            if int(item["index"]) not in key_set and cls._is_codec_residual_entry(item)
        ]

        if desired_count <= 0:
            return sorted(set(key_indices + residual_indices))
        if len(key_indices) >= desired_count:
            select = np.linspace(0, len(key_indices) - 1, desired_count, dtype=int).tolist()
            return sorted(set(key_indices[i] for i in select))

        need = max(int(desired_count) - len(key_indices), 0)
        if residual_indices and need > 0:
            if len(residual_indices) > need:
                select = np.linspace(0, len(residual_indices) - 1, need, dtype=int).tolist()
                residual_indices = [residual_indices[i] for i in select]
            else:
                residual_indices = residual_indices[:need]

        return sorted(set(key_indices + residual_indices))

    @staticmethod
    def _fallback_frame_type_label(frame):
        if frame_type_label_from_av is not None:
            return frame_type_label_from_av(frame)
        label = getattr(frame, "pict_type", None)
        label = str(label).split(".")[-1].upper() if label is not None else "UNKNOWN"
        if label == "UNKNOWN" and bool(getattr(frame, "key_frame", False)):
            label = "I"
        return label

    @staticmethod
    def _summarize_frame_type_labels(labels):
        if frame_type_summary is not None:
            return frame_type_summary(labels)
        counts = Counter(str(label).upper() for label in labels)
        return {
            "total": len(labels),
            "k_frames": counts.get("I", 0) + counts.get("IDR", 0) + counts.get("K", 0),
            "p_frames": counts.get("P", 0),
            "b_frames": counts.get("B", 0),
            "unknown_frames": counts.get("UNKNOWN", 0),
        }

    def _record_sampled_frame_types(self, resolved_path, frame_indices, decoded_frame_types=None, source="synthetic"):
        frame_indices = [int(idx) for idx in frame_indices]
        labels = None
        if decoded_frame_types is not None and len(decoded_frame_types) == len(frame_indices):
            labels = [str(label).upper() for label in decoded_frame_types]
            source = source or "decode"
        elif self._wants_codec_frame_types() and read_video_frame_types is not None and select_frame_type_labels is not None:
            try:
                max_frame = max(frame_indices) + 1 if frame_indices else None
                entries = read_video_frame_types(resolved_path, max_frames=max_frame)
                labels = select_frame_type_labels(entries, frame_indices)
                source = "codec"
            except Exception as e:
                eval_logger.warning(f"Failed to inspect codec frame types for {resolved_path}: {e}")

        if labels is None:
            labels = ["I" if pos == 0 else "P" for pos, _ in enumerate(frame_indices)]
            source = "synthetic"
        if labels:
            labels[0] = "I"

        self._last_frame_types = labels
        self._last_frame_indices = frame_indices
        self._last_frame_type_source = source
        self._last_frame_type_counts = self._summarize_frame_type_labels(labels)
        print(
            "[FRAME_TYPE_STATS] "
            f"total_frames={self._last_frame_type_counts.get('total', 0)} "
            f"k_frames={self._last_frame_type_counts.get('k_frames', 0)} "
            f"p_frames={self._last_frame_type_counts.get('p_frames', 0)} "
            f"b_frames={self._last_frame_type_counts.get('b_frames', 0)} "
            f"unknown_frames={self._last_frame_type_counts.get('unknown_frames', 0)} "
            f"source={source}",
            flush=True,
        )
        return labels

    def _frame_type_key_mask(self, fr: int, device, keep_flat=None):
        frame_types = self._last_frame_types
        if self._wants_codec_frame_types() and frame_types and len(frame_types) == fr:
            flags = []
            for label in frame_types:
                label = str(label).upper()
                if is_key_frame_type is not None:
                    flags.append(bool(is_key_frame_type(label)) or label == "K")
                else:
                    flags.append(label.startswith("I") or label == "K" or label == "IDR")
            if flags:
                flags[0] = True
            return torch.tensor(flags, dtype=torch.bool, device=device)

        key_frames = torch.zeros(fr, dtype=torch.bool, device=device)
        key_frames[0] = True
        if keep_flat is not None and torch.is_tensor(keep_flat) and keep_flat.dim() == 2 and keep_flat.shape[0] == fr:
            moved_counts = keep_flat.sum(dim=1)
            cuts = moved_counts > (self.scene_cut_factor * keep_flat.shape[1])
            key_frames[cuts] = True
        return key_frames

    def _frame_type_residual_mask(self, fr: int, device):
        frame_types = self._last_frame_types
        if self._wants_codec_frame_types() and frame_types and len(frame_types) == fr:
            flags = []
            for label in frame_types:
                label = str(label).upper()
                if is_p_frame_type is not None:
                    flags.append(bool(is_p_frame_type(label)))
                else:
                    flags.append(label.startswith(("P", "B")) or label in {"S", "SP", "BI"})
            if flags:
                flags[0] = False
            return torch.tensor(flags, dtype=torch.bool, device=device)

        residual_frames = torch.ones(fr, dtype=torch.bool, device=device)
        if fr > 0:
            residual_frames[0] = False
        return residual_frames

    def _get_motion_scores_for_scene_merge(self, out_feats, keep_flat=None):
        bf, n, _ = out_feats.shape
        scores = None
        try:
            gated = self._model.model.vision_tower.vision_tower.vision_model.embeddings
            scores = self._align_prune_aux_tensor(
                getattr(gated, "last_diff_score_flat", None),
                bf,
                n,
                out_feats.device,
                dtype=torch.float32,
            )
        except Exception:
            scores = None
        if scores is None:
            scores = self._align_prune_aux_tensor(keep_flat, bf, n, out_feats.device, dtype=torch.float32)
        if scores is None:
            scores = out_feats.float().norm(dim=-1)
        denom = scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
        return scores / denom

    def _record_scene_merge_metrics(self, applied: bool, before: int, after: int):
        self._last_scene_merge_applied = bool(applied)
        self._last_scene_merge_before_tokens = int(before)
        self._last_scene_merge_after_tokens = int(after)
        self._last_scene_merge_keep_ratio_actual = (float(after) / float(before)) if before else 1.0
        print(
            "[SCENE_MERGE_METRICS] "
            f"scene_merge_applied={str(bool(applied)).lower()} "
            f"scene_merge_before_tokens={before} "
            f"scene_merge_after_tokens={after} "
            f"scene_merge_keep_ratio_actual={self._last_scene_merge_keep_ratio_actual:.6f}",
            flush=True,
        )

    def _apply_scene_merge_packing(self, out_feats, keep_flat=None, kept_frame_idx=None):
        if not (
            self.use_vision_merge
            and self.scene_merge_apply
            and torch.is_tensor(out_feats)
            and out_feats.dim() == 3
        ):
            if torch.is_tensor(out_feats) and out_feats.dim() == 3:
                before = int(out_feats.shape[0] * out_feats.shape[1])
                self._record_scene_merge_metrics(False, before, before)
            return out_feats

        bf, n, d = out_feats.shape
        before = int(bf * n)
        ratio = self.scene_merge_keep_ratio
        if ratio is None:
            ratio = self._last_merge_ratio if self._last_merge_ratio is not None else 1.0
        ratio = min(max(float(ratio), 0.0), 1.0)
        if self.scene_merge_require_semantic_match and not self._last_scene_merge_has_semantic_match:
            self._record_scene_merge_metrics(False, before, before)
            self._last_merge_ratio = 1.0
            return out_feats
        k = max(int(self.scene_merge_min_tokens_per_frame), int(math.ceil(n * ratio)))
        k = min(max(k, 1), n)
        if k >= n:
            self._record_scene_merge_metrics(False, before, before)
            return out_feats

        key_mask = self._last_scene_keyframe_mask
        if key_mask is None or key_mask.numel() != bf:
            key_mask = self._frame_type_key_mask(bf, out_feats.device, keep_flat=keep_flat)

        kept_mask = torch.zeros(bf, dtype=torch.bool, device=out_feats.device)
        if kept_frame_idx is not None and torch.is_tensor(kept_frame_idx) and kept_frame_idx.numel() > 0:
            kept_mask[kept_frame_idx.long().clamp(min=0, max=bf - 1)] = True
        else:
            kept_mask = key_mask.clone()
        self._last_scene_kept_keyframe_mask = kept_mask

        motion_scores = self._get_motion_scores_for_scene_merge(out_feats, keep_flat=keep_flat)
        semantic_scores = out_feats.float().norm(dim=-1)
        semantic_scores = semantic_scores / semantic_scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
        scores = motion_scores.clone()

        full_key_rows = key_mask & kept_mask
        if full_key_rows.any():
            scores[full_key_rows] = semantic_scores[full_key_rows] + 1.0

        idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
        idx, _ = torch.sort(idx, dim=1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)
        merged = out_feats.gather(dim=1, index=gather_idx)
        after = int(bf * k)
        self._record_scene_merge_metrics(True, before, after)
        self._last_merge_ratio = self._last_scene_merge_keep_ratio_actual
        return merged

    # def summarize_stats(self):
    #     if not self.profiling:
    #         print("Profiling is disabled.")
    #         return

    #     s = self.stats
    #     total_samples = int(s["samples"])
    #     fps = s["frames_processed"] / s["vision_time"] if s["vision_time"]>0 else 0.0
    #     reduction = 1 - (s["kept_patches"] / s["orig_patches"] if s["orig_patches"]>0 else 0.0)

    #     print("—— Profiling Summary ——")
    #     print(f"Samples processed: {total_samples}")
    #     print(f"Total tokenization time: {s['tokenization_time']:.3f}s   avg per sample: {s['tokenization_time']/total_samples:.3f}s")
    #     print(f"Total LLM input tokens: {int(s['llm_input_tokens'])}   avg per sample: {s['llm_input_tokens']/total_samples:.1f}")
    #     print(f"Original patches: {int(s['orig_patches'])}, kept patches: {int(s['kept_patches'])}, reduction rate: {reduction:.2%}")
    #     print(f"Vision branch time: {s['vision_time']:.3f}s   avg per sample: {s['vision_time']/total_samples:.3f}s")
    #     print(f"Frames processed: {int(s['frames_processed'])}, throughput: {fps:.2f} fps")


    def _vision_merge_hook(self, module, inp, out_feats):
        """
        module   = mm_projector
        inp      = (...)
        out_feats: Tensor of shape [B*Fr, N, D]

        Important:
        LLaVA later splits encoded_image_features with split_sizes=[num_frames].
        So dim-0 must stay equal to the sampled frame count here.
        """
        gated = self._model.model.vision_tower.vision_tower.vision_model.embeddings
        keep_flat = getattr(gated, "last_keep_flat", None)
        if keep_flat is None:
            keep_flat = torch.ones(
                out_feats.shape[0], out_feats.shape[1], dtype=torch.bool, device=out_feats.device
            )

        bf, n, _ = out_feats.shape
        print(f"ori token number after tokenization is {bf * n}")

        if self.use_vision_merge:
            # Use frame-level pooled features only for merge decision/statistics.
            frame_feats = out_feats.mean(dim=1).unsqueeze(0)  # [1, Fr, D]
            kept_frame_idx = self._scene_token_merge(frame_feats, keep_flat)
            kept_frames = int(kept_frame_idx.numel())

            print(
                f"dense: scene_merge post_tokens_before={bf * n} "
                f"(split_safe, kept_frames={kept_frames}/{bf})\n"
            )
            out_feats = self._apply_scene_merge_packing(out_feats, keep_flat=keep_flat, kept_frame_idx=kept_frame_idx)
        else:
            self._last_merge_ratio = 1.0
            self._record_scene_merge_metrics(False, int(out_feats.shape[0] * out_feats.shape[1]), int(out_feats.shape[0] * out_feats.shape[1]))

        if self.enable_visual_token_pruning and self.prune_mode != "off":
            out_feats = self._apply_visual_token_pruning(out_feats, keep_flat=keep_flat)
        else:
            before = int(out_feats.shape[0] * out_feats.shape[1])
            self._record_prune_metrics(False, "off", self.prune_apply_mode, before, before)
            self._print_prune_metrics()

        if self.prune_mode == "codec_residual" and self.prune_apply_mode == "ragged_pack":
            self._last_post_tokens = int(self._last_post_tokens_after_prune or (out_feats.shape[0] * out_feats.shape[1]))
        else:
            self._last_post_tokens = int(out_feats.shape[0] * out_feats.shape[1])
        return out_feats


    def _scene_token_merge(self, feats: torch.Tensor, keep_flat: torch.BoolTensor) -> torch.Tensor:
        """
        feats:     [1, Fr, D]
        keep_flat: [Fr, N]
        return:    [1*Fr, D]
        """
        _, Fr, D = feats.shape
        key_frames = self._frame_type_key_mask(Fr, feats.device, keep_flat=keep_flat)
        self._last_scene_keyframe_mask = key_frames
        kf_idxs = key_frames.nonzero(as_tuple=False).squeeze(1)
        if kf_idxs.numel() < 2:
            self._last_merge_ratio = 1.0
            self._last_scene_merge_has_semantic_match = False
            print(f"scene merging ratio=1.000000 metric=jsd kept_frames={Fr}/{Fr}")
            return kf_idxs if kf_idxs.numel() > 0 else torch.arange(Fr, device=feats.device)

        def jsd(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (
                F.kl_div(p.log(), m, reduction="sum") +
                F.kl_div(q.log(), m, reduction="sum")
            )

        merged_feats = feats.clone()  # [1, Fr, D]
        cols_to_remove = []
        for i in range(len(kf_idxs) - 1):
            a, b = kf_idxs[i].item(), kf_idxs[i+1].item()
            pi, pj = merged_feats[0, a], merged_feats[0, b]  # [D]

            # semantic check
            dist = jsd(
                F.softmax(self.model.lm_head(pi), dim=-1),
                F.softmax(self.model.lm_head(pj), dim=-1),
            )
            # print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%{dist}%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            if dist >= self.jsd_threshold:
                
                continue

            # merge into the earlier slot
            merged_feats[0, a] = 0.5 * (pi + pj)
            cols_to_remove.append(b)
        keep = [i for i in range(merged_feats.size(1)) if i not in cols_to_remove]
        idx = torch.tensor(keep, device=merged_feats.device, dtype=torch.long)
        merged_feats_left = merged_feats.index_select(dim=1, index=idx)
        merge_ratio = (merged_feats_left.shape[0] * merged_feats_left.shape[1]) / (
            merged_feats.shape[0] * merged_feats.shape[1]
        )
        self._last_merge_ratio = float(merge_ratio)
        self._last_scene_merge_has_semantic_match = bool(cols_to_remove)
        print(
            f"scene merging ratio={merge_ratio:.6f} metric=jsd "
            f"kept_frames={merged_feats_left.shape[1]}/{Fr}"
        )
        return idx

    def merge_top2_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [N, D] —— 当前帧的 N 个 patch token 特征
        返回:   [N-1, D] —— 合并一次后剩下的 N-1 个 token
        """
        N, D = tokens.shape

        # 1) 标准化到单位向量，方便做余弦相似度
        norms = tokens.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = tokens / norms

        # 2) 计算两两余弦相似度矩阵
        sim = normed @ normed.T           # [N, N]
        sim.fill_diagonal_(-1.0)          # 屏蔽自己

        # 3) 找到相似度最大的那一对 i,j
        idx_flat = sim.view(-1).argmax()
        i = idx_flat // N
        j = idx_flat % N

        # 4) 计算新 token
        new_token = 0.5 * (tokens[i] + tokens[j])  # [D]

        # 5) 删除原来两个 token，拼回剩下的 + 新 token
        mask = torch.ones(N, dtype=torch.bool, device=tokens.device)
        mask[i] = False
        mask[j] = False
        remaining = tokens[mask]             # [N-2, D]
        merged = torch.cat([remaining, new_token.unsqueeze(0)], dim=0)  # [N-1, D]

        return merged


    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    # —— 在所有 tokenizer_image_token 调用处打点 —— 
    def _tokenize_and_track(self, prompt: str):
        ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return ids

    # —— 替换原有调用 —— 
    # input_ids = tokenizer_image_token(...)
    # 改为：
    # input_ids = self._tokenize_and_track(prompt)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num, dense_frame_fps=self.dense_frame_fps)
                        elif self.video_decode_backend == "pyav":
                            frames = self.load_video_pyav(visual, self.max_frames_num, dense_frame_fps=self.dense_frame_fps)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self._tokenize_and_track(prompt=prompt).unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = self._tokenize_and_track(prompt=full_prompt).unsqueeze(0).to(self.device)

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            with torch.inference_mode():
                outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    @staticmethod
    def _is_cuda_oom_error(err: Exception) -> bool:
        text = str(err).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def _apply_oom_adaptive_caps(self, observed_tokens: Optional[int], observed_frames: Optional[int]) -> bool:
        changed = False

        if observed_tokens is not None and observed_tokens > 0:
            est_cap = max(self.min_token_cap, int(observed_tokens * self.oom_token_reduce_ratio))
        elif self._dynamic_visual_token_cap is not None:
            est_cap = max(self.min_token_cap, int(self._dynamic_visual_token_cap * self.oom_token_reduce_ratio))
        else:
            est_cap = self.min_token_cap

        if self._dynamic_visual_token_cap is None or est_cap < self._dynamic_visual_token_cap:
            self._dynamic_visual_token_cap = est_cap
            changed = True

        if observed_frames is not None and observed_frames > 0:
            est_frame_cap = max(self.min_frame_cap, int(observed_frames * self.oom_frame_reduce_ratio))
        elif self._dynamic_max_frames_num is not None:
            est_frame_cap = max(self.min_frame_cap, int(self._dynamic_max_frames_num * self.oom_frame_reduce_ratio))
        else:
            est_frame_cap = self.min_frame_cap

        if self._dynamic_max_frames_num is None or est_frame_cap < self._dynamic_max_frames_num:
            self._dynamic_max_frames_num = est_frame_cap
            changed = True

        eval_logger.warning(
            f"[OOM_GUARD] adapt caps -> max_frames={self._dynamic_max_frames_num}, "
            f"token_cap={self._dynamic_visual_token_cap}"
        )
        print(
            f"[OOM_GUARD] adapt caps max_frames={self._dynamic_max_frames_num} "
            f"token_cap={self._dynamic_visual_token_cap}",
            flush=True,
        )
        return changed

    def _prepare_video_tensor(self, visual):
        last_err = None
        for attempt in range(3):
            try:
                if self.video_decode_backend == "decord":
                    try:
                        frames = self.load_video(visual, self.max_frames_num, dense_frame_fps=self.dense_frame_fps)
                    except Exception as e:
                        eval_logger.warning(f"Decord decode failed; falling back to pyav. Error: {e}")
                        num_frm = self._dynamic_max_frames_num if self._dynamic_max_frames_num is not None else self.max_frames_num
                        frames = self.load_video_pyav(visual, num_frm, dense_frame_fps=self.dense_frame_fps)
                elif self.video_decode_backend == "pyav":
                    num_frm = self._dynamic_max_frames_num if self._dynamic_max_frames_num is not None else self.max_frames_num
                    frames = self.load_video_pyav(visual, num_frm, dense_frame_fps=self.dense_frame_fps)
                else:
                    raise ValueError(f"Unsupported video_decode_backend: {self.video_decode_backend}")
                break
            except Exception as e:
                last_err = e
                if attempt >= 2:
                    raise
                eval_logger.warning(f"Video decode retry {attempt + 1}/2 after error: {e}")
                time.sleep(0.5 * (attempt + 1))
        else:
            raise last_err

        pixel_values = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensor = [pixel_values]
        return frames, image_tensor

    def load_video_pyav(self, video_path, max_frames_num, dense_frame_fps=None):
        if isinstance(video_path, str):
            resolved_path = video_path
        else:
            resolved_path = video_path[0]
        resolved_path = resolve_video_path(resolved_path)

        container = av.open(resolved_path)
        try:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            orig_fps = float(stream.average_rate or 0.0)
            if orig_fps <= 0:
                orig_fps = 30.0

            if total_frames <= 0 and stream.duration is not None and stream.time_base is not None:
                total_frames = int(float(stream.duration * stream.time_base) * orig_fps)
            if total_frames <= 0:
                raise ValueError(f"PyAV stream frame count unavailable: {resolved_path}")

            if self.clip_duration_sec is not None and self.clip_duration_sec > 0:
                capped_frames = min(total_frames, max(int(orig_fps * self.clip_duration_sec), 1))
            else:
                capped_frames = total_frames

            duration_sec = capped_frames / orig_fps if orig_fps > 0 else 0.0
            strategy = (self.frame_sampling_strategy or "uniform").lower()
            dense_fps = dense_frame_fps if dense_frame_fps is not None else self.dense_frame_fps

            if dense_fps is not None and dense_fps > 0:
                desired_count = max(int(round(duration_sec * float(dense_fps))), 1)
            else:
                desired_count = int(max_frames_num) if max_frames_num is not None else capped_frames

            if strategy == "stride":
                if dense_fps is not None and dense_fps > 0:
                    step = max(int(round(orig_fps / float(dense_fps))), 1)
                else:
                    step = self.frame_stride
                frame_indices = list(range(0, capped_frames, step))
            elif strategy in {"codec_keyframe", "codec_iframe", "codec_kp", "codec_ip"}:
                codec_entries = []
                idx = 0
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if idx >= capped_frames:
                            break
                        label = self._fallback_frame_type_label(frame)
                        is_key = bool(getattr(frame, "key_frame", False)) or label.startswith("I")
                        is_residual = (not is_key) and (label.startswith(("P", "B")) or label in {"S", "SP", "BI"})
                        codec_entries.append(
                            {
                                "index": idx,
                                "pict_type": label,
                                "is_keyframe": is_key,
                                "is_p_frame": is_residual,
                            }
                        )
                        idx += 1
                    if idx >= capped_frames:
                        break
                frame_indices = self._select_codec_key_or_residual_indices(
                    codec_entries,
                    desired_count=desired_count,
                    include_residual=strategy in {"codec_kp", "codec_ip"},
                )
                container.close()
                container = av.open(resolved_path)
            elif strategy == "keyframe":
                key_indices = []
                idx = 0
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if idx >= capped_frames:
                            break
                        if frame.key_frame:
                            key_indices.append(idx)
                        idx += 1
                    if idx >= capped_frames:
                        break
                if not key_indices:
                    key_indices = [0]
                frame_indices = sorted(set(key_indices))
                if desired_count > 0 and len(frame_indices) > desired_count:
                    select = np.linspace(0, len(frame_indices) - 1, desired_count, dtype=int).tolist()
                    frame_indices = [key_indices[i] for i in select]
                container.close()
                container = av.open(resolved_path)
            else:
                sample_count = min(max(desired_count, 1), capped_frames)
                frame_indices = np.linspace(0, capped_frames - 1, sample_count, dtype=int).tolist()

            hard_cap = None
            if max_frames_num is not None and max_frames_num > 0:
                hard_cap = int(max_frames_num)
            if self._dynamic_max_frames_num is not None and self._dynamic_max_frames_num > 0:
                hard_cap = min(hard_cap, int(self._dynamic_max_frames_num)) if hard_cap is not None else int(self._dynamic_max_frames_num)
            if self.high_fps_guard and dense_fps is not None and dense_fps >= self.high_fps_threshold_fps and self.high_fps_max_frames > 0:
                hard_cap = min(hard_cap, int(self.high_fps_max_frames)) if hard_cap is not None else int(self.high_fps_max_frames)

            if hard_cap is not None and len(frame_indices) > hard_cap:
                before = len(frame_indices)
                select = np.linspace(0, len(frame_indices) - 1, hard_cap, dtype=int).tolist()
                frame_indices = [frame_indices[i] for i in select]
                print(
                    f"[OOM_GUARD] frame_truncated before={before} after={len(frame_indices)} cap={hard_cap}",
                    flush=True,
                )

            if self.force_include_last_frame and (capped_frames - 1) not in frame_indices:
                frame_indices.append(capped_frames - 1)

            frame_indices = sorted(set(int(i) for i in frame_indices if 0 <= int(i) < capped_frames))
            if hard_cap is not None and len(frame_indices) > hard_cap:
                select = np.linspace(0, len(frame_indices) - 1, hard_cap, dtype=int).tolist()
                frame_indices = [frame_indices[i] for i in select]
                if self.force_include_last_frame and frame_indices and frame_indices[-1] != (capped_frames - 1):
                    frame_indices[-1] = capped_frames - 1
                frame_indices = sorted(set(frame_indices))
            if not frame_indices:
                frame_indices = [0]

            frames_array = None
            # The Decord build used by some GRT environments can fail against
            # the host FFmpeg.  For a full-video uniform sample, seek directly
            # to the eight requested timestamps instead of decoding every frame
            # up to the end of a multi-hour video.
            if strategy == "uniform" and capped_frames == total_frames:
                try:
                    frames_array = read_video_pyav_seek_uniform(
                        resolved_path,
                        num_frm=len(frame_indices),
                        fps=None,
                        format="rgb24",
                        force_include_last_frame=self.force_include_last_frame,
                    )
                    if len(frames_array) != len(frame_indices):
                        raise ValueError(
                            f"Seek decoder returned {len(frames_array)} frames; expected {len(frame_indices)}"
                        )
                    self._record_sampled_frame_types(resolved_path, frame_indices, source="seek")
                except Exception as exc:
                    eval_logger.warning(f"PyAV seek sampling failed; using sequential decode. Error: {exc}")
                    frames_array = None

            if frames_array is None:
                requested = set(frame_indices)
                end_index = max(requested)
                frames = []
                decoded_types = {}
                for idx, frame in enumerate(container.decode(video=0)):
                    if idx > end_index:
                        break
                    if idx in requested:
                        frames.append(frame.to_ndarray(format="rgb24"))
                        decoded_types[idx] = self._fallback_frame_type_label(frame)

                if len(frames) != len(requested):
                    raise ValueError(
                        f"Decoded {len(frames)} of {len(requested)} requested frames from {resolved_path}"
                    )
                sampled_types = [decoded_types.get(idx, "I" if pos == 0 else "P") for pos, idx in enumerate(frame_indices)]
                self._record_sampled_frame_types(resolved_path, frame_indices, decoded_frame_types=sampled_types, source="decode")
                frames_array = np.stack(frames)

            sampled = len(frame_indices)
            effective_fps = sampled / duration_sec if duration_sec > 0 else 0.0
            self._last_sampled_frames = int(sampled)
            self._last_effective_fps = float(effective_fps)
            self._last_video_duration = float(duration_sec)

            print(
                f"[FPS_STATS] strategy={strategy} target_fps={dense_fps} "
                f"orig_fps={orig_fps:.3f} duration_s={duration_sec:.3f} "
                f"total_frames={total_frames} capped_frames={capped_frames} "
                f"sampled_frames={sampled} effective_fps={effective_fps:.3f}",
                flush=True,
            )

            return frames_array
        finally:
            container.close()

    def load_video(self, video_path, max_frames_num, dense_frame_fps=None):
        if isinstance(video_path, str):
            resolved_path = video_path
        else:
            resolved_path = video_path[0]
        resolved_path = resolve_video_path(resolved_path)

        vr = VideoReader(resolved_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Empty video: {resolved_path}")

        orig_fps = float(vr.get_avg_fps() or 0.0)
        if orig_fps <= 0:
            orig_fps = 30.0

        # Optional truncation by clip duration to make high-FPS runs feasible and comparable.
        if self.clip_duration_sec is not None and self.clip_duration_sec > 0:
            capped_frames = min(total_frames, max(int(orig_fps * self.clip_duration_sec), 1))
        else:
            capped_frames = total_frames

        duration_sec = capped_frames / orig_fps if orig_fps > 0 else 0.0

        strategy = (self.frame_sampling_strategy or "uniform").lower()
        dense_fps = dense_frame_fps if dense_frame_fps is not None else self.dense_frame_fps

        # Desired frame count for uniform/keyframe strategies.
        if dense_fps is not None and dense_fps > 0:
            desired_count = max(int(round(duration_sec * float(dense_fps))), 1)
        else:
            desired_count = int(max_frames_num) if max_frames_num is not None else capped_frames

        if strategy == "stride":
            if dense_fps is not None and dense_fps > 0:
                step = max(int(round(orig_fps / float(dense_fps))), 1)
            else:
                step = self.frame_stride
            frame_indices = list(range(0, capped_frames, step))
        elif strategy in {"codec_keyframe", "codec_iframe", "codec_kp", "codec_ip"}:
            if read_video_frame_types is None:
                raise RuntimeError(f"{strategy} sampling requires PyAV codec frame inspection support.")
            entries = read_video_frame_types(resolved_path, max_frames=capped_frames)
            frame_indices = self._select_codec_key_or_residual_indices(
                entries,
                desired_count=desired_count,
                include_residual=strategy in {"codec_kp", "codec_ip"},
            )
        elif strategy == "keyframe":
            # Keyframe baseline: decode key frames and map pts->index.
            container = av.open(resolved_path)
            key_indices = []
            idx = 0
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    if idx >= capped_frames:
                        break
                    if frame.key_frame:
                        key_indices.append(idx)
                    idx += 1
                if idx >= capped_frames:
                    break
            container.close()
            if not key_indices:
                key_indices = [0]
            frame_indices = sorted(set(key_indices))
            if desired_count > 0 and len(frame_indices) > desired_count:
                frame_indices = np.linspace(0, len(frame_indices) - 1, desired_count, dtype=int).tolist()
                frame_indices = [key_indices[i] for i in frame_indices]
        else:
            # uniform
            sample_count = min(max(desired_count, 1), capped_frames)
            frame_indices = np.linspace(0, capped_frames - 1, sample_count, dtype=int).tolist()

        # Optional hard cap to avoid OOM if caller sets an extremely high FPS.
        hard_cap = None
        if max_frames_num is not None and max_frames_num > 0:
            hard_cap = int(max_frames_num)
        if self._dynamic_max_frames_num is not None and self._dynamic_max_frames_num > 0:
            hard_cap = min(hard_cap, int(self._dynamic_max_frames_num)) if hard_cap is not None else int(self._dynamic_max_frames_num)
        if self.high_fps_guard and dense_fps is not None and dense_fps >= self.high_fps_threshold_fps and self.high_fps_max_frames > 0:
            hard_cap = min(hard_cap, int(self.high_fps_max_frames)) if hard_cap is not None else int(self.high_fps_max_frames)

        if hard_cap is not None and len(frame_indices) > hard_cap:
            before = len(frame_indices)
            select = np.linspace(0, len(frame_indices) - 1, hard_cap, dtype=int).tolist()
            frame_indices = [frame_indices[i] for i in select]
            print(
                f"[OOM_GUARD] frame_truncated before={before} after={len(frame_indices)} cap={hard_cap}",
                flush=True,
            )

        if self.force_include_last_frame and (capped_frames - 1) not in frame_indices:
            frame_indices.append(capped_frames - 1)

        frame_indices = sorted(set(int(i) for i in frame_indices if 0 <= int(i) < capped_frames))
        if hard_cap is not None and len(frame_indices) > hard_cap:
            select = np.linspace(0, len(frame_indices) - 1, hard_cap, dtype=int).tolist()
            frame_indices = [frame_indices[i] for i in select]
            if self.force_include_last_frame and frame_indices and frame_indices[-1] != (capped_frames - 1):
                frame_indices[-1] = capped_frames - 1
            frame_indices = sorted(set(frame_indices))
        if not frame_indices:
            frame_indices = [0]

        frames = vr.get_batch(frame_indices).asnumpy()
        self._record_sampled_frame_types(resolved_path, frame_indices)

        sampled = len(frame_indices)
        effective_fps = sampled / duration_sec if duration_sec > 0 else 0.0
        self._last_sampled_frames = int(sampled)
        self._last_effective_fps = float(effective_fps)
        self._last_video_duration = float(duration_sec)

        print(
            f"[FPS_STATS] strategy={strategy} target_fps={dense_fps} "
            f"orig_fps={orig_fps:.3f} duration_s={duration_sec:.3f} "
            f"total_frames={total_frames} capped_frames={capped_frames} "
            f"sampled_frames={sampled} effective_fps={effective_fps:.3f}",
            flush=True,
        )

        return frames  # (frames, H, W, C)

    # def load_video(self, video_path, max_frames_num, dense_frame_fps=None):
    #     """
    #     Load frames from a video, either by:
    #     1) uniform sampling of up to max_frames_num frames, or
    #     2) sampling at an approximate target FPS (sample_fps).

    #     Args:
    #     video_path (str or list): path to the video file (or a list whose first
    #                                 element is the path).
    #     max_frames_num (int): maximum number of frames to return when sample_fps is None.
    #     dense_frame_fps (float, optional): if set, sample frames at ~this FPS,
    #                                     ignoring max_frames_num.

    #     Returns:
    #     np.ndarray: an array of frames, shape (n_frames, height, width, channels).
    #     """




    #     # 1. Open the video
    #     if isinstance(video_path, str):
    #         vr = VideoReader(video_path, ctx=cpu(0))
    #     else:
    #         vr = VideoReader(video_path[0], ctx=cpu(0))

    #     # 2. Get total number of frames and the video's original FPS
    #     total_frames = len(vr)
    #     orig_fps = vr.get_avg_fps()  # e.g., 30.0

    #     if dense_frame_fps is not None:
    #         # 3a. Compute integer step size to approximate target FPS
    #         #    (orig_fps / sample_fps) gives # of original frames per sampled frame
    #         step = max(int(orig_fps / dense_frame_fps), 1)

    #         # 4a. Build a list of indices every `step` frames
    #         frame_indices = list(range(0, total_frames, step))
    #     else:
    #         # 3b. Uniformly sample `max_frames_num` indices across the entire video
    #         frame_indices = np.linspace(
    #             0, total_frames - 1, max_frames_num, dtype=int
    #         ).tolist()

    #     # 5. Fetch the selected frames in one batch
    #     dense_frames = vr.get_batch(frame_indices).asnumpy()

    #     orig_fps = vr.get_avg_fps()  # e.g. 30.0    
    #     sampled = len(dense_frames)
    #     effective_fps = sampled * orig_fps / total_frames
    #     print(f"[load_video] sampled {sampled}/{total_frames} frames → effective {effective_fps:.2f} FPS", flush=True)


    #     return dense_frames  # shape = (n_selected, height, width, channels)

    # def load_video(self, video_path, max_frames_num, dense_frame_fps=None):
    #     """
    #     Load frames from a video, either by:
    #     1) uniform sampling of up to max_frames_num frames, or
    #     2) sampling at an approximate target FPS (sample_fps).

    #     Args:
    #     video_path (str or list): path to the video file (or a list whose first
    #                                 element is the path).
    #     max_frames_num (int): maximum number of frames to return when sample_fps is None.
    #     dense_frame_fps (float, optional): if set, sample frames at ~this FPS,
    #                                     ignoring max_frames_num.

    #     Returns:
    #     np.ndarray: an array of frames, shape (n_frames, height, width, channels).
    #     """




    #     # 1. Open the video
    #     if isinstance(video_path, str):
    #         vr = VideoReader(get_video_path(video_path), ctx=cpu(0))
    #     else:
    #         vr = VideoReader(get_video_path(video_path[0]), ctx=cpu(0))

    #     # 2. Get total number of frames and the video's original FPS
    #     total_frames = len(vr)
    #     orig_fps = vr.get_avg_fps()  # e.g., 30.0

    #     if dense_frame_fps is not None:
    #         # 3a. Compute integer step size to approximate target FPS
    #         #    (orig_fps / sample_fps) gives # of original frames per sampled frame
    #         step = max(int(orig_fps / dense_frame_fps), 1)

    #         # 4a. Build a list of indices every `step` frames
    #         frame_indices = list(range(0, total_frames, step))
    #     else:
    #         # 3b. Uniformly sample `max_frames_num` indices across the entire video
    #         frame_indices = np.linspace(
    #             0, total_frames - 1, max_frames_num, dtype=int
    #         ).tolist()

    #     # 5. Fetch the selected frames in one batch
    #     dense_frames = vr.get_batch(frame_indices).asnumpy()

    #     orig_fps = vr.get_avg_fps()  # e.g. 30.0    
    #     sampled = len(dense_frames)
    #     effective_fps = sampled * orig_fps / total_frames
    #     print(f"[load_video] sampled {sampled}/{total_frames} frames → effective {effective_fps:.2f} FPS", flush=True)


    #     return dense_frames  # shape = (n_selected, height, width, channels)


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                self._last_pre_tokens = None
                self._last_post_tokens = None
                self._last_tokenization_time = None
                self._last_gate_keep_ratio = None
                self._last_recomputed_patches = None
                self._last_orig_patches = None
                self._last_recompute_ratio = None
                self._last_gate_policy = "disabled"
                self._last_gate_metric = self.gate_metric
                self._last_merge_ratio = None
                self._last_sampled_frames = None
                self._last_effective_fps = None
                self._last_video_duration = None
                self._last_frame_types = None
                self._last_frame_indices = None
                self._last_frame_type_counts = {}
                self._last_frame_type_source = "none"
                self._last_ragged_token_mask = None
                self._last_scene_merge_applied = False
                self._last_scene_merge_before_tokens = None
                self._last_scene_merge_after_tokens = None
                self._last_scene_merge_keep_ratio_actual = 1.0
                self._last_scene_merge_has_semantic_match = False
                self._last_scene_keyframe_mask = None
                self._last_scene_kept_keyframe_mask = None

                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        # 1) load frames
                        t0 = time.perf_counter()
                        image_tensor = []
                        try:
                            frames, image_tensor = self._prepare_video_tensor(visual)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            raise

                        task_type = "video"
                        placeholder_count = len(frames) if (frames is not None and self.token_strategy == "multiple") else 1
                        # 2) preprocess
                        t1 = time.perf_counter()

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [self._tokenize_and_track(prompt=prompt) for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            run_wall_start = time.perf_counter()
            last_err = None
            cont = None
            for attempt in range(self.oom_retry_times + 1):
                try:
                    with torch.inference_mode():
                        if self.profiling:
                            t2 = time.perf_counter()
                            cont = self.model.generate(
                                input_ids,
                                attention_mask=attention_masks,
                                pad_token_id=pad_token_ids,
                                images=image_tensor,
                                use_cache=self.use_cache,
                                **gen_kwargs,
                            )
                            t3 = time.perf_counter()

                            if "frames" in locals() and frames is not None:
                                num_frames = frames.shape[0]
                                self.stats["frames_processed"] += num_frames
                            self.stats["vision_time"] += (t3 - t2)
                            if "t1" in locals() and "t0" in locals():
                                self.stats["load_preprocess_time"] += (t1 - t0)
                        else:
                            cont = self.model.generate(
                                input_ids,
                                attention_mask=attention_masks,
                                pad_token_id=pad_token_ids,
                                images=image_tensor,
                                use_cache=self.use_cache,
                                **gen_kwargs,
                            )
                    last_err = None
                    break
                except RuntimeError as e:
                    last_err = e
                    if task_type != "video" or not self._is_cuda_oom_error(e) or attempt >= self.oom_retry_times:
                        raise e

                    observed_tokens = None
                    if isinstance(self._last_post_tokens, int) and self._last_post_tokens > 0:
                        observed_tokens = self._last_post_tokens
                    elif isinstance(self._last_pre_tokens, int) and self._last_pre_tokens > 0:
                        observed_tokens = self._last_pre_tokens

                    observed_frames = None
                    if "frames" in locals() and frames is not None:
                        try:
                            observed_frames = int(frames.shape[0])
                        except Exception:
                            observed_frames = None

                    changed = self._apply_oom_adaptive_caps(observed_tokens=observed_tokens, observed_frames=observed_frames)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not changed:
                        raise e

                    try:
                        frames, image_tensor = self._prepare_video_tensor(visual)
                    except Exception as reload_e:
                        eval_logger.error(f"[OOM_GUARD] failed to rebuild video tensor after OOM: {reload_e}")
                        raise e

                    eval_logger.warning(
                        f"[OOM_GUARD] OOM retry {attempt + 1}/{self.oom_retry_times} "
                        f"with max_frames={self._dynamic_max_frames_num}, token_cap={self._dynamic_visual_token_cap}"
                    )
                    print(
                        f"[OOM_GUARD] retry={attempt + 1} max_frames={self._dynamic_max_frames_num} "
                        f"token_cap={self._dynamic_visual_token_cap}",
                        flush=True,
                    )
                    continue

            if last_err is not None:
                raise last_err

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            run_wall_end = time.perf_counter()

            text_outputs = [response.strip() for response in text_outputs]

            if task_type == "video":
                wall_time_s = run_wall_end - run_wall_start
                sampled_frames = (
                    self._last_sampled_frames
                    if self._last_sampled_frames is not None
                    else (len(frames) if ("frames" in locals() and frames is not None) else 0)
                )
                throughput_fps = (sampled_frames / wall_time_s) if wall_time_s > 0 and sampled_frames else 0.0

                pre_tokens = self._last_pre_tokens if self._last_pre_tokens is not None else 0
                post_tokens = self._last_post_tokens if self._last_post_tokens is not None else 0
                retention_ratio = (post_tokens / pre_tokens) if pre_tokens else 0.0
                gate_keep_ratio = self._last_gate_keep_ratio if self._last_gate_keep_ratio is not None else 1.0
                recomputed_patches = self._last_recomputed_patches if self._last_recomputed_patches is not None else pre_tokens
                orig_patches = self._last_orig_patches if self._last_orig_patches is not None else pre_tokens
                recompute_ratio = self._last_recompute_ratio if self._last_recompute_ratio is not None else 1.0
                gate_policy = self._last_gate_policy or ("motion" if gate_keep_ratio < 1.0 else "disabled")
                gate_metric = self._last_gate_metric or self.gate_metric
                merge_ratio = self._last_merge_ratio if self._last_merge_ratio is not None else 1.0
                tokenization_time_s = self._last_tokenization_time if self._last_tokenization_time is not None else 0.0
                effective_fps = self._last_effective_fps if self._last_effective_fps is not None else 0.0
                pruning_enabled = bool(self._last_pruning_enabled)
                prune_mode = self._last_prune_mode or "off"
                prune_apply_mode = self._last_prune_apply_mode or self.prune_apply_mode
                post_tokens_before_prune = (
                    self._last_post_tokens_before_prune
                    if self._last_post_tokens_before_prune is not None
                    else post_tokens
                )
                post_tokens_after_prune = (
                    self._last_post_tokens_after_prune
                    if self._last_post_tokens_after_prune is not None
                    else post_tokens
                )
                prune_keep_ratio_actual = (
                    self._last_prune_keep_ratio_actual
                    if self._last_prune_keep_ratio_actual is not None
                    else 1.0
                )
                scene_merge_applied = bool(self._last_scene_merge_applied)
                scene_merge_before_tokens = (
                    self._last_scene_merge_before_tokens
                    if self._last_scene_merge_before_tokens is not None
                    else post_tokens
                )
                scene_merge_after_tokens = (
                    self._last_scene_merge_after_tokens
                    if self._last_scene_merge_after_tokens is not None
                    else post_tokens
                )
                scene_merge_keep_ratio_actual = self._last_scene_merge_keep_ratio_actual
                frame_type_counts = self._last_frame_type_counts or {}

                print(
                    "[DENSE_METRICS] "
                    f"sampled_frames={sampled_frames} "
                    f"effective_fps={effective_fps:.6f} "
                    f"wall_time_s={wall_time_s:.6f} "
                    f"throughput_fps={throughput_fps:.6f} "
                    f"pre_tokens={pre_tokens} "
                    f"post_tokens={post_tokens} "
                    f"retention_ratio={retention_ratio:.6f} "
                    f"pruning_enabled={str(pruning_enabled).lower()} "
                    f"prune_mode={prune_mode} "
                    f"prune_apply_mode={prune_apply_mode} "
                    f"post_tokens_before_prune={post_tokens_before_prune} "
                    f"post_tokens_after_prune={post_tokens_after_prune} "
                    f"prune_keep_ratio_actual={prune_keep_ratio_actual:.6f} "
                    f"scene_merge_applied={str(scene_merge_applied).lower()} "
                    f"scene_merge_before_tokens={scene_merge_before_tokens} "
                    f"scene_merge_after_tokens={scene_merge_after_tokens} "
                    f"scene_merge_keep_ratio_actual={scene_merge_keep_ratio_actual:.6f} "
                    f"codec_k_frames={frame_type_counts.get('k_frames', 0)} "
                    f"codec_p_frames={frame_type_counts.get('p_frames', 0)} "
                    f"codec_b_frames={frame_type_counts.get('b_frames', 0)} "
                    f"codec_unknown_frames={frame_type_counts.get('unknown_frames', 0)} "
                    f"frame_type_source={self._last_frame_type_source} "
                    f"gate_keep_ratio={gate_keep_ratio:.6f} "
                    f"recomputed_patches={recomputed_patches} "
                    f"orig_patches={orig_patches} "
                    f"recompute_ratio={recompute_ratio:.6f} "
                    f"gate_policy={gate_policy} "
                    f"gate_metric={gate_metric} "
                    f"merge_ratio={merge_ratio:.6f} "
                    f"tokenization_time_s={tokenization_time_s:.6f}",
                    flush=True,
                )

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_to_text, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # multi round inference: terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []

                if round_idx != 0:  # get current round visual and context from doc_to_text function
                    batched_visuals, batched_contexts, batched_terminal_singal, batched_round_res, batched_previous_round_info = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    # import ipdb; ipdb.set_trace()
                    batched_round_res = list(zip(*batched_round_res))  # [(r1_1, r1_2), (r2_1, r2_2), ...]
                    if batched_terminal_singal[0]:  # terminal signal from doc_to_text function
                        break

                for visual, context in zip(batched_visuals, batched_contexts):
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == "pyav":
                                    frames = self.load_video_pyav(visual, self.max_frames_num, dense_frame_fps=self.dense_frame_fps)
                                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                        """
                        # if task_type == "image": # indeed in multi-image case, not the video in frames.
                        #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        # elif task_type == "video":
                        # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()

                    if utils.is_json(question):  # conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [self._tokenize_and_track(prompt=prompt) for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if task_type == "image":
                    gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == "video":
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs["modalities"] = ["video"]
                    gen_kwargs["stopping_criteria"] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs.keys():
                    gen_kwargs.pop("image_aspect_ratio")
                try:
                    with torch.inference_mode():
                        cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e

                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial("generate_until_multi_round", (context, gen_kwargs), batched_round_res)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
