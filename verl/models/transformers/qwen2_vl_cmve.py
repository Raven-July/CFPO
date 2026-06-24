# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on:
# https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, List, Union
import math
import torch

from ...utils.py_functional import is_transformers_version_greater_than
from .flash_attention_utils import flash_attention_forward


if is_transformers_version_greater_than("4.52.0"):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLAttention,
        Qwen2VLCausalLMOutputWithPast,
        Qwen2VLForConditionalGeneration,
        Qwen2VLModel,
        Qwen2VLModelOutputWithPast,
        Qwen2VLTextModel,
        Qwen2VLDecoderLayer,
        BaseModelOutputWithPast,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )
    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from transformers.cache_utils import DynamicCache
    from transformers.utils import logging

    logger = logging.get_logger(__name__)
else:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLAttention,
        Qwen2VLCausalLMOutputWithPast,
        Qwen2VLForConditionalGeneration,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )


def get_rope_index(
    processor: "Qwen2VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1405
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device
        )  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            )
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
            )

    return position_ids


def add_diffusion_noise(image_tensor, noise_step, gamma=0.005):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (gamma - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    # import pdb;pdb.set_trace()

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    return image_tensor_cd


def GMM_mask(
    sig,
    thres_mode,
    valid_mask=None,  # 只统计有效部分
):
    data = sig

    # 如果没有传入 valid_mask，默认全部非零元素有效，或者全部有效
    if valid_mask is None:
        # 简单的做法是认为 > 0 的才是有效的 attention (排除了 casual mask 和 sample packing 的 mask)
        valid_mask = data > 1e-6

    # 提取有效数据用于统计计算
    valid_data = data[valid_mask]

    # 防止全是0的情况导致 NaN
    if valid_data.numel() == 0:
        return torch.zeros_like(data), torch.tensor(0.0), torch.tensor(0.0)

    # 全局统计
    mean = torch.mean(valid_data)
    std = torch.std(valid_data)

    if thres_mode == "high":
        thres = mean + 2 * std
    elif thres_mode == "medium":
        thres = mean + std
    elif thres_mode == "low":
        thres = mean
    elif thres_mode == "extra":
        thres = mean + 3 * std
    else:
        raise ValueError(f"Unknown thres_mode: {thres_mode}")

    # 生成 mask (基于原始 data 形状)
    mask = (sig > thres).float()

    # 再次应用 valid_mask，确保被 Mask 掉的区域（如 Padding 或 其他样本）强制为 0
    if valid_mask is not None:
        mask = mask * valid_mask.float()
    # 计算sig > thres的比例
    # ratio = torch.mean((sig > thres).float()).item()
    # print(f"比例 (sig > thres): {ratio:.4f} ({ratio*100:.2f}%)")
    return mask, mean, std


def values_noise_multiply(
    attn_weights, image_pos, value_states, thres_mode, noise, mean_mode
):
    bsz, num_heads, q_len, k_len = attn_weights.shape
    head_dim = value_states.shape[-1]

    if noise:
        noise_value_states = add_diffusion_noise(value_states.clone(), 999, 0.01)
        # print("actually added noise!")
    else:
        # 根据 mean_mode 初始化 mask
        if mean_mode == "text":
            # text模式：初始全为True (假设全为text)，遇到image位置置为False
            value_mask = torch.ones(
                (bsz, k_len), dtype=torch.bool, device=value_states.device
            )
            # print("mean_mode: text")
        elif mean_mode == "image":
            # image模式：初始全为False，遇到image位置置为True
            value_mask = torch.zeros(
                (bsz, k_len), dtype=torch.bool, device=value_states.device
            )
            # print("mean_mode: image")
        else:
            raise ValueError(f"Unknown mean_mode: {mean_mode}")
        num_tokens_per_item = torch.zeros(
            (bsz, 1), dtype=value_states.dtype, device=value_states.device
        )
        for i, intervals in enumerate(image_pos):
            count = 0
            for interval in intervals:
                start, end = interval["image_token_start"], interval["image_token_end"]
                if start < k_len and end <= k_len:
                    # 根据模式更新 mask
                    if mean_mode == "text":
                        value_mask[i, start:end] = False
                    else:
                        value_mask[i, start:end] = True
                        # print("value_mask: image")
            # 统计有效 token 数量
            count = value_mask[i].sum()
            num_tokens_per_item[i] = max(count, 1.0)

        value_mask_expanded = value_mask.view(bsz, 1, k_len, 1).expand_as(value_states)
        masked_values = value_states.where(
            value_mask_expanded,
            torch.tensor(0.0, dtype=value_states.dtype, device=value_states.device),
        )
        mean = torch.sum(masked_values, dim=(2, 3), keepdim=True) / (
            num_tokens_per_item.view(bsz, 1, 1, 1) * head_dim
        )
        noise_value_states = mean.expand_as(value_states)
    # --------------------------------------------------------
    final_mask = torch.zeros_like(attn_weights)

    for i, intervals in enumerate(image_pos):
        for interval in intervals:
            start, end = interval["image_token_start"], interval["image_token_end"]

            if q_len > end and start < end:
                # 提取切片
                img_attn_slice = attn_weights[i : i + 1, :, end:, start:end]

                if img_attn_slice.numel() > 0:
                    # 在 Causal Attention 机制下，如果 img_attn_slice 为 0 (或极小值)，
                    # 说明这些位置是被 Mask 掉的（可能是 Padding，可能是未来的 token，也可能是 Packed 的其他样本）
                    # 只对 > 0 的部分计算 GMM 统计量

                    # 使用一个极小的阈值 epsilon 防止 float 误差
                    valid_slice_mask = img_attn_slice > 1e-8

                    # 只有当切片内有有效值时才计算，否则全0
                    if valid_slice_mask.any():
                        mask, _, _ = GMM_mask(
                            img_attn_slice,
                            valid_mask=valid_slice_mask,
                            thres_mode=thres_mode,
                        )
                        final_mask[i : i + 1, :, end:, start:end] = mask

    final_mask = final_mask.to(value_states.dtype)
    return final_mask, noise_value_states


# Add Cross-Modal-Value-Enhancement(CMVE)
def qwen2_vl_attn_forward(
    self: "Qwen2VLAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    apply_cmve: bool = False,  # <-- 新增标志（默认 False）
    image_pos: Optional[List[dict]] = None,  # <-- 新增参数，图像位置
    thres_mode: str = None,  # <-- 新增参数，阈值模式
    noise: bool = False,  # <-- 新增参数，是否添加扩散噪声
    mean_mode: str = None,  # <-- 新增参数，均值计算模式
    **kwargs,
) -> Tuple[torch.Tensor, None, None]:
    # print("qwen2_vl_attn_forward:" + str(apply_cmve))
    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    # (batch_size, seq_length / sp_size, num_heads * head_size)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    if apply_cmve:
        # calculate
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(
                torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights
            )

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        final_mask, ave_value_states = values_noise_multiply(
            attn_weights, image_pos, value_states, thres_mode, noise, mean_mode
        )

        delta_v = ave_value_states - value_states
        attn_output_delta = torch.matmul(attn_weights * final_mask, delta_v)
        if attn_output_delta.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output_delta` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output_delta.size()}"
            )
        attn_output_delta = attn_output_delta.transpose(1, 2)

        attn_output_ori, _ = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            position_ids=position_ids[0],  # important: pass position ids
        )  # (batch_size, seq_length, num_head / sp_size, head_size)

        attn_output = attn_output_ori + attn_output_delta
    else:
        attn_output, _ = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            position_ids=position_ids[0],  # important: pass position ids
        )  # (batch_size, seq_length, num_head / sp_size, head_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None


def _get_input_embeds(
    model: "Qwen2VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        pixel_values = torch.zeros(
            (16, 1176), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        image_grid_thw = torch.tensor(
            [[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device
        )
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def qwen2_vl_base_forward_new(
    self: "Qwen2VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    apply_cmve: bool = False,  # <-- 新增标志（默认 False）
    image_pos: Optional[List[dict]] = None,  # <-- 新增参数，图像位置
    thres_mode: str = None,  # <-- 新增参数，阈值模式
    noise: bool = False,  # <-- 新增参数，是否添加扩散噪声
    mean_mode: str = None,  # <-- 新增参数，均值计算模式
    **kwargs,
):
    # print("qwen2_vl_base_forward_new:" + str(apply_cmve))
    inputs_embeds, attention_mask = _get_input_embeds(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
    )
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        apply_cmve=apply_cmve,  # <-- 传递标志
        image_pos=image_pos,
        thres_mode=thres_mode,
        noise=noise,
        mean_mode=mean_mode,
        **kwargs,
    )

    return Qwen2VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=None,
    )


def qwen2_vl_forward_new(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    thres_mode: str = None,  # <-- 新增参数，阈值模式
    noise: bool = False,  # <-- 新增参数，是否添加扩散噪声
    mean_mode: str = None,  # <-- 新增参数，均值计算模式
    **kwargs,
) -> "Qwen2VLCausalLMOutputWithPast":
    print("THIS IS CMVE FORWARD")
    print(
        "Using thres_mode:",
        thres_mode,
        " Using noise:",
        noise,
        " Using mean_mode:",
        mean_mode,
    )

    image_pos = []  # 结构为 List[List[dict]]

    # 遍历批处理中的每个输入序列
    for i in range(input_ids.shape[0]):
        sample_intervals = []

        # 获取所有 vision_start 和 vision_end 的索引
        bos_indices = torch.where(input_ids[i] == self.config.vision_start_token_id)[0]
        eos_indices = torch.where(input_ids[i] == self.config.vision_end_token_id)[0]

        # 检查数量是否匹配 (通常应该是成对出现的)
        if len(bos_indices) == len(eos_indices) and len(bos_indices) > 0:
            # 遍历每一对 start/end
            for start_idx, end_idx in zip(bos_indices, eos_indices):
                s = start_idx.item() + 1
                e = end_idx.item()

                # 简单的合法性检查
                if s < e:
                    sample_intervals.append(
                        {
                            "image_token_start": s,
                            "image_token_end": e,
                        }
                    )
        else:
            # 异常情况处理：数量不匹配或没有图片
            if len(bos_indices) > 0:
                min_len = min(len(bos_indices), len(eos_indices))
                for k in range(min_len):
                    sample_intervals.append(
                        {
                            "image_token_start": bos_indices[k].item() + 1,
                            "image_token_end": eos_indices[k].item(),
                        }
                    )
        # print(sample_intervals)

        image_pos.append(sample_intervals)

    outputs_mod = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        apply_cmve=True,  # <-- 触发变化 A
        image_pos=image_pos,  # <-- 传递图像位置
        thres_mode=thres_mode,  # <-- 可以调整阈值模式
        noise=noise,
        mean_mode=mean_mode,
        **kwargs,
    )
    hidden_states_cf = outputs_mod[0]
    logits_cf = self.lm_head(hidden_states_cf)

    return Qwen2VLCausalLMOutputWithPast(
        loss=None,
        logits=logits_cf,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        rope_deltas=None,
    )


def qwen2_vl_language_forward_new(
    self: "Qwen2VLTextModel",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    apply_cmve: bool = False,  # <-- 新增标志（默认 False）
    image_pos: Optional[List[dict]] = None,  # <-- 新增参数，图像位置
    thres_mode: str = None,  # <-- 新增参数，阈值模式
    noise: bool = False,  # <-- 新增参数，是否添加扩散噪声
    mean_mode: str = None,  # <-- 新增参数，均值计算模式
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # print("qwen2_vl_language_forward_new:" + str(apply_cmve))
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                apply_cmve,  # <-- 传递标志
                image_pos,
                thres_mode,
                noise,
                mean_mode,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                apply_cmve=apply_cmve,  # <-- 传递标志
                image_pos=image_pos,
                thres_mode=thres_mode,
                noise=noise,
                mean_mode=mean_mode,
                **kwargs,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_vl_decoder_forward_new(
    self: "Qwen2VLDecoderLayer",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    apply_cmve: bool = False,  # <-- 新增标志（默认 False）
    image_pos: Optional[List[dict]] = None,  # <-- 新增参数，图像位置
    thres_mode: str = None,  # <-- 新增参数，阈值模式
    noise: bool = False,  # <-- 新增参数，是否添加扩散噪声
    mean_mode: str = None,  # <-- 新增参数，均值计算模式
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # print("qwen2_vl_decoder_forward_new:" + str(apply_cmve))

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        apply_cmve=apply_cmve,  # <-- 传递标志
        image_pos=image_pos,  # <-- 传递图像位置
        thres_mode=thres_mode,  # <-- 传递阈值模式
        noise=noise,  # <-- 传递是否添加噪声
        mean_mode=mean_mode,  # <-- 传递均值计算模式
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
