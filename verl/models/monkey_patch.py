# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.py_functional import is_transformers_version_greater_than
from .transformers.flash_attention_utils import flash_attention_forward


def make_forward_with_default(func, thres_mode="high", noise=False, mean_mode="image"):
    def wrapper(self, *args, **kwargs):
        # 如果用户没有传 a，就用默认值
        if "thres_mode" not in kwargs:
            kwargs["thres_mode"] = thres_mode
        if "noise" not in kwargs:
            kwargs["noise"] = noise
        if "mean_mode" not in kwargs:
            kwargs["mean_mode"] = mean_mode
        return func(self, *args, **kwargs)

    return wrapper


def apply_ulysses_patch(
    model_type: str, apply_cmve: bool, thres_mode="high", noise=False, mean_mode="image"
) -> None:

    if model_type in (
        "llama",
        "gemma",
        "gemma2",
        "mistral",
        "qwen2",
        "qwen3",
        "qwen3_moe",
    ):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        if apply_cmve == False:
            from .transformers.qwen2_vl import (
                qwen2_vl_attn_forward,
                qwen2_vl_base_forward_new,
                qwen2_vl_forward_new,
            )

            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration,
                Qwen2VLModel,
            )

            Qwen2VLModel.forward = qwen2_vl_base_forward_new
            Qwen2_5_VLModel.forward = qwen2_vl_base_forward_new
            Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward_new
            Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_forward_new
        else:
            from .transformers.qwen2_vl_cmve import (
                qwen2_vl_attn_forward,
                qwen2_vl_base_forward_new,
                qwen2_vl_forward_new,
                qwen2_vl_language_forward_new,
                qwen2_vl_decoder_forward_new,
            )

            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
                Qwen2_5_VLTextModel,
                Qwen2_5_VLDecoderLayer,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration,
                Qwen2VLModel,
                Qwen2VLTextModel,
                Qwen2VLDecoderLayer,
            )

            Qwen2VLModel.forward = qwen2_vl_base_forward_new
            Qwen2_5_VLModel.forward = qwen2_vl_base_forward_new
            Qwen2VLForConditionalGeneration.forward = make_forward_with_default(
                qwen2_vl_forward_new,
                thres_mode=thres_mode,
                noise=noise,
                mean_mode=mean_mode,
            )
            Qwen2_5_VLForConditionalGeneration.forward = make_forward_with_default(
                qwen2_vl_forward_new,
                thres_mode=thres_mode,
                noise=noise,
                mean_mode=mean_mode,
            )
            Qwen2VLTextModel.forward = qwen2_vl_language_forward_new
            Qwen2_5_VLTextModel.forward = qwen2_vl_language_forward_new
            Qwen2VLDecoderLayer.forward = qwen2_vl_decoder_forward_new
            Qwen2_5_VLDecoderLayer.forward = qwen2_vl_decoder_forward_new

        if is_transformers_version_greater_than("4.53.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLAttention,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLAttention,
            )

            Qwen2VLAttention.forward = qwen2_vl_attn_forward
            Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLFlashAttention2,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLFlashAttention2,
            )

            Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
            Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward
    else:
        raise NotImplementedError(
            f"Model architecture {model_type} is not supported yet."
        )
