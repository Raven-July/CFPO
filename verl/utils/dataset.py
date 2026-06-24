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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        if image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        # --- MODIFICATION: Changed data_path type hint ---
        data_path: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        image_dir: Optional[str] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.image_dir = image_dir  # Global fallback
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        # --- MODIFICATION: Reworked dataset loading logic ---
        all_datasets = []
        if isinstance(data_path, list):
            # New logic: use the provided list
            dataset_configs = data_path
        else:
            raise ValueError("data_path must be a list of dictionaries.")

        for config in dataset_configs:
            current_data_path = config["dataset"]
            # Use .get() for 'image_path', allowing it to be missing
            current_image_path = config.get("image")

            if "@" in current_data_path:
                current_data_path, data_split = current_data_path.split("@")
            else:
                data_split = "train"

            if os.path.isdir(current_data_path):
                file_type = os.path.splitext(os.listdir(current_data_path)[0])[-1][
                    1:
                ].replace("jsonl", "json")
                loaded_ds = load_dataset(
                    file_type, data_dir=current_data_path, split=data_split
                )
            elif os.path.isfile(current_data_path):
                file_type = os.path.splitext(current_data_path)[-1][1:].replace(
                    "jsonl", "json"
                )
                loaded_ds = load_dataset(
                    file_type, data_files=current_data_path, split=data_split
                )
            else:
                # load remote dataset from huggingface hub
                loaded_ds = load_dataset(current_data_path, split=data_split)

            # Add the per-dataset image path as a new column
            # Use a "private" name to avoid conflicts
            if current_image_path is not None:
                # Use num_proc=16 for consistency with filter logic
                loaded_ds = loaded_ds.map(
                    lambda example: {"__internal_image_path__": current_image_path},
                    num_proc=16,
                )
            all_datasets.append(loaded_ds)

        if not all_datasets:
            # Handle case where data_path was an empty list
            raise ValueError("No datasets were loaded. data_path was empty or invalid.")

        if len(all_datasets) > 1:
            from datasets import concatenate_datasets

            self.dataset = concatenate_datasets(all_datasets)
        else:
            self.dataset = all_datasets[0]
        # --- END MODIFICATION ---

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=32,
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = self.format_prompt.strip()
            # format_prompt = Template(self.format_prompt.strip())
            # prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [
                {
                    "role": "system",
                    "content": format_prompt,
                },
                {"role": "user", "content": content_list},
            ]
        else:
            return [
                {
                    "role": "system",
                    "content": format_prompt,
                },
                {"role": "user", "content": prompt_str},
            ]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            images = example[self.image_key] or []

            # --- MODIFICATION: Get per-sample image path or fallback to global ---
            current_image_dir = example["__internal_image_path__"]
            # --- END MODIFICATION ---

            if (
                current_image_dir is not None  # <-- MODIFIED
                and len(images) != 0
                and isinstance(images[0], str)
            ):  # image paths
                images = [
                    os.path.join(current_image_dir, image) for image in images
                ]  # <-- MODIFIED

            resized_images = [
                process_image(
                    image, min_pixels=self.min_pixels, max_pixels=self.max_pixels
                )
                for image in images
            ] or None
            model_inputs = self.processor(
                resized_images, [prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            images = example.pop(self.image_key)

            # --- MODIFICATION: Get per-sample image path or fallback to global ---
            current_image_dir = example["__internal_image_path__"]
            # --- END MODIFICATION ---
            if (
                current_image_dir is not None  # <-- MODIFIED
                and len(images) != 0
                and isinstance(images[0], str)
            ):  # image paths
                images = [
                    os.path.join(current_image_dir, image) for image in images
                ]  # <-- MODIFIED

            resized_images = [
                process_image(
                    image, min_pixels=self.min_pixels, max_pixels=self.max_pixels
                )
                for image in images
            ] or None
            model_inputs = self.processor(
                resized_images, [prompt], add_special_tokens=False, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(
                [prompt], add_special_tokens=False, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1, min=0, max=None
            )  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        answer_val = example.pop(self.answer_key)

        # 动态判断并组装 ground_truth
        # tensor形状通常是 (num_images, 3)，取第一张图转为 list: [t, h, w]
        if "solution" in example:
            # print("rec mission!")
            example["ground_truth"] = {
                "answer": answer_val,
                "solution": example.pop("solution"),
                # 这里假设前面已经处理好了 images 和 model_inputs
                "image_path": images if (current_image_dir is not None and len(images) != 0) else [],
                "image_grid_thw": model_inputs.get("image_grid_thw")[0].tolist() if model_inputs.get("image_grid_thw") is not None else []
            }
        else:
            example["ground_truth"] = answer_val

        return example
