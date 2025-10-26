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


class ImageResizeTool(BaseTool):
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
        logger.info(f"Initialized ImageResizeTool with config: {config}")

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

        img = [{"image": images[idx]} for idx in range(len(images))]

        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        instance_data = self._instance_dict[instance_id]
        images = instance_data["image"]

        timestamps = parameters.get("frame_indices")

        if timestamps is None or not isinstance(timestamps, list) or len(timestamps) == 0:
            return (
                ToolResponse(text="Error: frame_indices parameter is missing or not a list of integers."),
                -0.05,
                {"success": False},
            )

        invalid = [t for t in timestamps if not isinstance(t, int) or t < 0 or t >= len(images)]
        if invalid:
            return (
                ToolResponse(text=f"Error: frame_indices contains out-of-range indices: {invalid}. The valid range is [0, {len(images)-1}]."),
                -0.05,
                {"success": False},
            )

        if len(timestamps) > 5:
            return (
                ToolResponse(text="Error: Too many frame indices. Please select at most 5 frame indices."),
                -0.05,
                {"success": False},
            )

        try:
            selected_images = [images[idx]["image"] for idx in timestamps]
        except Exception as e:
            logger.error(f"Error processing image resize: {e}")
            return ToolResponse(text=f"Error processing image resize: {e}"), -0.05, {"success": False}

        response_text = f"Selected images at frame indices {timestamps}."

        with open('/workspace/log/tool.log', 'a') as f:
            f.write(f'response_text: {response_text}\n')
            f.write(f'selected_images: {len(selected_images)} {type(selected_images[0])}\n')

        return (
            ToolResponse(
                image=selected_images,
                text=response_text,
            ),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# # /home/geng/git/verl/verl/tools/image_resize_tool.py
# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import json
# import logging
# import os
# import threading
# from contextlib import ExitStack
# from enum import Enum
# from typing import Any, Callable, Optional, TypeVar
# from uuid import uuid4

# import ray
# import ray.actor
# from qwen_vl_utils import fetch_image

# from .base_tool import BaseTool
# from .schemas import OpenAIFunctionToolSchema, ToolResponse

# logger = logging.getLogger(__name__)
# logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# T = TypeVar("T")


# # Adapted from verl/tools/sandbox_fusion_tools.py
# class PoolMode(Enum):
#     """Execution pool mode enumeration."""

#     ThreadMode = 1
#     ProcessMode = 2


# @ray.remote(concurrency_groups={"acquire": 1, "release": 10})
# class TokenBucketWorker:
#     """Ray actor for rate limiting using token bucket algorithm."""

#     def __init__(self, rate_limit: int):
#         self.rate_limit = rate_limit
#         self.current_count = 0  # For observability
#         self._semaphore = threading.Semaphore(rate_limit)

#     @ray.method(concurrency_group="acquire")
#     def acquire(self):
#         """Acquire a token from the bucket."""
#         self._semaphore.acquire()
#         self.current_count += 1

#     @ray.method(concurrency_group="release")
#     def release(self):
#         """Release a token back to the bucket."""
#         self._semaphore.release()
#         self.current_count -= 1

#     def get_current_count(self):
#         """Get current number of acquired tokens."""
#         return self.current_count


# class VisualExecutionWorker:
#     """Worker for executing visual processing operations with optional rate limiting."""

#     def __init__(self, enable_global_rate_limit=True, rate_limit=10):
#         self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

#     def _init_rate_limit(self, rate_limit):
#         """Initialize singleton rate limiter."""
#         return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

#     def ping(self):
#         """Health check method."""
#         return True

#     def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
#         """Execute function with optional rate limiting."""
#         if self.rate_limit_worker:
#             with ExitStack() as stack:
#                 stack.callback(self.rate_limit_worker.release.remote)
#                 ray.get(self.rate_limit_worker.acquire.remote())
#                 try:
#                     return fn(*fn_args, **fn_kwargs)
#                 except Exception as e:
#                     # TODO we should make this available to the tool caller
#                     logger.warning(f"Error when executing visual processing: {e}")
#         else:
#             return fn(*fn_args, **fn_kwargs)


# def init_visual_execution_pool(
#     num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
# ):
#     """Initialize visual execution pool."""
#     if mode == PoolMode.ThreadMode:
#         return (
#             ray.remote(VisualExecutionWorker)
#             .options(max_concurrency=num_workers)
#             .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
#         )
#     else:
#         raise NotImplementedError("Process mode is not implemented yet")


# class ImageResizeTool(BaseTool):
#     """A tool for resizing selected images from an image sequence.

#     This tool accepts an image sequence at creation time and resizes the
#     images specified by `indexes` during execution. Target size is taken
#     from tool config.

#     Methods:
#         get_openai_tool_schema: Return the tool schema in OpenAI format
#         create: Create a tool instance for a trajectory (loads image sequence)
#         execute: Resize images at given `indexes` using target size
#         calc_reward: Calculate the reward with respect to tool state
#         release: Release the tool instance
#     """

#     def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
#         """
#         _tool_schema = OpenAIFunctionToolSchema.model_validate({
#             "type": "function",
#             "function": {
#                 "name": "image_resize_tool",
#                 "description": "Resize selected images from a preloaded image sequence by index.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "indexes": {
#                             "type": "array",
#                             "items": {"type": "integer"},
#                             "minItems": 1,
#                             "description": "List of indexes within the preloaded image sequence to resize."
#                         }
#                     },
#                     "required": ["indexes"],
#                 },
#             }
#         })
#         """
#         super().__init__(config, tool_schema)
#         self._instance_dict = {}

#         # Worker and rate limiting configuration
#         self.num_workers = config.get("num_workers", 20)
#         self.rate_limit = config.get("rate_limit", 50)
#         self.timeout = config.get("timeout", 30)

#         # Target size for resizing (required)
#         _default_size = 28 ** 2
#         self.target_width = int(config.get("target_width", _default_size))
#         self.target_height = int(config.get("target_height", _default_size))

#         self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
#         self.execution_pool = init_visual_execution_pool(
#             num_workers=self.num_workers,
#             enable_global_rate_limit=self.enable_global_rate_limit,
#             rate_limit=self.rate_limit,
#             mode=PoolMode.ThreadMode,
#         )
#         logger.info(f"Initialized ImageResizeTool with config: {config}")

#     def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
#         return self.tool_schema

#     async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
#         """
#         Creates a new instance for image resize tool, loading an image sequence.

#         Args:
#             instance_id: Optional unique identifier for the instance. If not provided, a new UUID is generated.
#             **kwargs: Should contain 'images' or 'image_sequence' with a list of image-like inputs. Each element can be:
#                 - A PIL.Image.Image object.
#                 - A string containing an HTTP or HTTPS URL.
#                 - A string containing a local file path.
#                 - A string containing a file URI (e.g., "file:///path/to/image.jpg").
#                 - A string containing a base64-encoded image "data:image/jpeg;base64,..."

#         Returns:
#             Tuple of (instance_id, ToolResponse)
#         """
#         if instance_id is None:
#             instance_id = str(uuid4())

#         # Handle create_kwargs parameter if passed
#         create_kwargs = kwargs.get("create_kwargs", {})
#         if create_kwargs:
#             kwargs.update(create_kwargs)

#         # Get image sequence from kwargs
#         images_arg = kwargs.get("images", None)
#         if images_arg is None:
#             images_arg = kwargs.get("image_sequence", None)

#         if images_arg is None or not isinstance(images_arg, list) or len(images_arg) == 0:
#             raise ValueError("Missing required 'images' (list) parameter in kwargs")

#         # Load/normalize each image item using fetch_image
#         # loaded_images = []
#         # for item in images_arg:

#         #     with open('/workspace/log/tool.log', 'a') as f:
#         #         f.write(f'item: {item}\n')
#         #         f.write('image loaded!!!!\n')

#         #     img = fetch_image({"image": item})
#         #     loaded_images.append(img)

#         loaded_images = images_arg

#         with open('/workspace/log/tool.log', 'a') as f:
#             f.write(f'loaded_images: {len(loaded_images)} {type(loaded_images[0])}\n')
#             f.write('images loaded!!!!\n')

#         self._instance_dict[instance_id] = {
#             "images": loaded_images,
#             "response": "",
#             "reward": 0.0,
#         }
#         return instance_id, ToolResponse()

#     async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:

#         with open('/workspace/log/tool.log', 'a') as f:
#             f.write('function is called!!!!!!\n')
#             f.write(json.dumps(parameters))
#             f.write('\n')

#         indexes = parameters.get("timestamps")

#         # Validate indexes
#         if indexes is None or not isinstance(indexes, list) or len(indexes) == 0:
#             return (
#                 ToolResponse(text="Error: 'timestamps' parameter is required and must be a non-empty list of integers."),
#                 -0.05,
#                 {"success": False},
#             )

#         try:
#             idx_list = [int(i) for i in indexes]
#         except Exception:
#             return (
#                 ToolResponse(text="Error: All 'timestamps' must be integers."),
#                 -0.05,
#                 {"success": False},
#             )

#         # remove duplicates
#         idx_list = list(set(idx_list))

#         if len(idx_list) > 4:
#             return (
#                 ToolResponse(text="Error: Too many timestamps. Please select at most 4 timestamps."),
#                 -0.05,
#                 {"success": False},
#             )

#         instance_data = self._instance_dict[instance_id]
#         images = instance_data["images"]
#         n = len(images)

#         # Range check
#         for i in idx_list:
#             if i < 0 or i >= n:
#                 return (
#                     ToolResponse(text=f"Error: timestamp {i} is out of range [0, {n-1}]."),
#                     -0.05,
#                     {"success": False},
#                 )

#         # Perform resizing
#         try:
#             # resized_images = [images[i].resize((self.target_width, self.target_height)) for i in idx_list]
#             resized_images = [images[i] for i in idx_list]
#             logger.info(f"Resized {len(resized_images)} images to: {self.target_width}x{self.target_height}")
#         except Exception as e:
#             logger.error(f"Error resizing images: {e}")
#             return ToolResponse(text=f"Error resizing images: {e}"), -0.05, {"success": False}

#         response_text = f"Resized images at timestamps {idx_list} to {self.target_width}x{self.target_height}."

#         with open('/workspace/log/tool.log', 'a') as f:
#             f.write(f'response_text: {response_text}\n')
#             f.write('response_text!!!!\n')

#         return (
#             ToolResponse(
#                 image=resized_images,
#                 text=response_text,
#             ),
#             0.0,
#             {"success": True, "count": len(resized_images)},
#         )

#     async def release(self, instance_id: str, **kwargs) -> None:
#         if instance_id in self._instance_dict:
#             del self._instance_dict[instance_id]