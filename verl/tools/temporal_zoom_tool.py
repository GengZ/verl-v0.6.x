# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import re
import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
from PIL import Image

import ray
import ray.actor
from qwen_vl_utils import fetch_image
from verl.utils.dataset.vision_utils import process_image, process_video

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


# def compute_tool_reward(raw_text: str) -> float:
#     text = raw_text or ""
#     reward = 0.0

#     # Find inner JSONs inside <tool_call>...</tool_call>
#     inner_matches = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text)
#     open_count = text.count("<tool_call>")
#     close_count = text.count("</tool_call>")

#     # 1) exactly one pair
#     if len(inner_matches) == 1 and open_count == 1 and close_count == 1:
#         reward += 0.5

#     # 2) short text
#     if len(text) < 500:
#         reward += 0.5

#     # 3) first match: name == image_resize_tool and <= 5 frame_indices
#     if inner_matches:
#         for first_payload in inner_matches:
#             try:
#                 obj = json.loads(first_payload.strip())
#                 if obj.get("name") == "image_resize_tool":
#                     args = obj.get("arguments", {})
#                     frames = args.get("frame_indices")
#                     if isinstance(frames, list) and len(frames) <= 5:
#                         reward += 0.5
#                         break
#             except Exception:
#                 pass

#     return reward


class TemporalZoomTool(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box.

    This tool provides a zoom-in functionality by cropping a region from an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the zoom-in operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional).",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized TemporalZoomTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image zoom-in tool.

        This method initializes a new session for an image, which can then be used
        for operations like zooming. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        images = kwargs.get("images")
        if images is None:
            raise ValueError("Missing required 'images' parameter in kwargs")

        self._instance_dict[instance_id] = {
            "images": images,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # NOTE: the images input is a list of images, but we need to select a sublist of images with length LENGTH
        #       it DOES NOT support pass in dict of images
        LENGTH = 8

        instance_data = self._instance_dict[instance_id]
        images = instance_data["images"]

        interval_index = parameters.get("interval_index")

        # raw_text = parameters.get("_agent_raw_text")
        # tool_reward = compute_tool_reward(raw_text)
        
        if isinstance(interval_index, str):
            try:
                interval_index = int(interval_index)
            except Exception as e:
                logger.error(f"Error converting interval_index to integer: {e}")
                return (
                    ToolResponse(text=f"Error: interval_index is not an integer: {interval_index}."),
                    -0.05,
                    {"success": False},
                )

        if interval_index is None or not isinstance(interval_index, int):
            return (
                ToolResponse(text="Error: interval_index parameter is missing or not an integer."),
                -0.05,
                {"success": False},
            )

        invalid = interval_index < 0 or interval_index >= int(len(images) / LENGTH)
        if invalid:
            return (
                ToolResponse(text=f"Error: interval_index is an out-of-range index: {interval_index}. The valid range is [0, {int(len(images) / LENGTH)-1}]."),
                -0.05,
                {"success": False},
            )

        try:
            idx_start = interval_index * LENGTH
            idx_end = idx_start + LENGTH

            selected_images = images[idx_start:idx_end]

            if not isinstance(selected_images, list):
                selected_images = [selected_images]
        except Exception as e:
            logger.error(f"Error Retrieving Images: {e}")
            return ToolResponse(text=f"Error Retrieving Images: {e}"), -0.05, {"success": False}

        response_text = f"Zoomed in on the frames between Frame-{interval_index} and Frame-{interval_index+1}."

        # with open('/workspace/log/tool.log', 'a') as f:
        #     f.write(f'response_text: {response_text}\n')
        #     f.write(f'selected_images: {len(selected_images)} {type(selected_images[0])}\n')

        return (
            ToolResponse(
                image=selected_images,
                text=response_text,
            ),
            1.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
