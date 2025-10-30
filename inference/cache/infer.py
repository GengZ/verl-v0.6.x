#!/usr/bin/env python3
import os
import re
import cv2
import json
import shutil
import tempfile
import numpy as np
import torch
from itertools import combinations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import decord

@dataclass
class Config:
    model_path: str
    video_path: str
    question: str
    correct_answer: Optional[str] = None
    work_dir: Optional[str] = None
    device_str: Optional[str] = None
    max_iterations: int = 5
    max_retries: int = 3
    num_frames_to_sample: int = 8
    num_frames_to_sample_long: int = 12
    max_frame_width: int = 640
    max_frame_height: int = 360
    max_frame_width_long: int = 448
    max_frame_height_long: int = 252
    long_video_threshold_s: int = 300

def load_model_and_processor(model_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def run_inference(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.9)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text

def parse_model_response(response):
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    return (think_match.group(1).strip(), action_match.group(1).strip()) if think_match and action_match else (None, None)

def get_video_metadata(video_path):
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        frame_count = len(vr)
        del vr
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps, frame_count
    except Exception:
        return 0, 0

def scale_down_preserving_aspect_ratio(image, max_width=640, max_height=360):
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    ratio_w = max_width / w
    ratio_h = max_height / h
    scale_ratio = min(ratio_w, ratio_h)
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_frames(video_path, frame_indices, output_dir, max_width, max_height):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        frames_array = vr.get_batch(frame_indices).asnumpy()
        for i, frame_idx in enumerate(frame_indices):
            frame_img_bgr = cv2.cvtColor(frames_array[i], cv2.COLOR_RGB2BGR)
            scaled_frame = scale_down_preserving_aspect_ratio(frame_img_bgr, max_width=max_width, max_height=max_height)
            output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, scaled_frame)
            saved_paths.append(output_path)
        del vr
    except Exception as e:
        print("!!!!!! AN EXCEPTION OCCURRED in extract_frames !!!!!!")
        print(f"Video Path: {video_path}")
        print(f"Frame Indices: {frame_indices}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
    return saved_paths

def handle_get_frame_number(action_content, fps):
    time_match = re.match(r'get frame number at time\s+(\S+)', action_content.strip())
    if not time_match:
        return None
    try:
        minutes, seconds = map(int, time_match.group(1).split(':'))
        return f"Frame number at time {time_match.group(1)} is: {int((minutes * 60 + seconds) * fps)}."
    except ValueError:
        return None

def handle_choose_frames(action_content, video_path, video_id, iteration, total_frames, base_frame_dir,
                         num_frames_to_sample, max_width, max_height):
    match = re.search(r"choose frames between\s+(\d+)\s+and\s+(\d+)", action_content)
    if not match:
        return None
    try:
        start_frame, end_frame = map(int, match.groups())
    except ValueError:
        return None
    if start_frame >= end_frame or end_frame >= total_frames or (end_frame - start_frame) <= num_frames_to_sample:
        return None
    frame_indices = np.linspace(start_frame + 1, end_frame - 1, num_frames_to_sample, dtype=int).tolist()
    output_dir = os.path.join(base_frame_dir, f"{video_id}_iter_{iteration}")
    image_paths = extract_frames(video_path, frame_indices, output_dir, max_width, max_height)
    user_content = []
    for path, index in zip(image_paths, frame_indices):
        user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])
    return user_content

def parse_frame_number(filepath: str) -> int:
    match = re.search(r'_(\d+)\.jpg$', filepath)
    return int(match.group(1)) if match else -1

def stringify_conversation(conversation_history: list, num_frames_to_sample: int) -> str:
    image_paths = []
    for turn in conversation_history:
        if isinstance(turn.get("content"), list):
            for item in turn["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_paths.append(item.get("image"))
    image_groups = [image_paths[i:i + num_frames_to_sample] for i in range(0, len(image_paths), num_frames_to_sample)]
    frame_intervals = []
    for group in image_groups:
        if not group:
            continue
        start_frame_num = parse_frame_number(group[0])
        end_frame_num = parse_frame_number(group[-1])
        if start_frame_num != -1 and end_frame_num != -1:
            frame_intervals.append((min(start_frame_num, end_frame_num), max(start_frame_num, end_frame_num)))
    if len(frame_intervals) > 1:
        for interval1, interval2 in combinations(frame_intervals, 2):
            s1, e1 = interval1
            s2, e2 = interval2
            if s1 == s2 and e1 == e2:
                return "error"
            if abs(s1 - s2) <= 1 and abs(e1 - e2) <= 1:
                return "error"
    stringified_content = "".join([
        turn['content'] if isinstance(turn.get('content'), str) else "\n[New frames provided.]\n"
        for turn in conversation_history[2:]
    ])
    return stringified_content

def validate_reasoning_process(predict_str: str, num_frames_to_sample: int):
    try:
        think_contents = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        action_contents = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)
        system_responses = re.findall(r'</action>(.*?)(?:<think>|$)', predict_str, re.DOTALL)
    except Exception:
        return False, 0, 0
    if not think_contents or not action_contents or len(think_contents) != len(action_contents):
        return False, 0, 0
    if not action_contents[-1].strip().startswith('output answer:'):
        return False, 0, 0
    tool_call_count = len(action_contents) - 1
    image_add_count = 0
    action_frame_pairs, requested_times = [], []
    expected_frame_in_next_action = None
    for i, action in enumerate(action_contents[:-1]):
        action = action.strip()
        if expected_frame_in_next_action is not None:
            frame_match_check = re.match(r'choose frames between (\d+) and (\d+)', action)
            if not frame_match_check:
                return False, 0, 0
            start_f, end_f = map(int, frame_match_check.groups())
            if not (start_f <= expected_frame_in_next_action <= end_f):
                return False, 0, 0
            expected_frame_in_next_action = None
        frame_match = re.match(r'choose frames between (\d+) and (\d+)', action)
        if frame_match:
            image_add_count += 1
            num1, num2 = map(int, frame_match.groups())
            current_pair = (num1, num2)
            if current_pair in action_frame_pairs:
                return False, 0, 0
            action_frame_pairs.append(current_pair)
            if num1 >= num2 - num_frames_to_sample:
                return False, 0, 0
            continue
        time_match = re.match(r'get frame number at time\s+(\S+)', action)
        if time_match:
            time_str_action = time_match.group(1)
            if time_str_action in requested_times:
                return False, 0, 0
            requested_times.append(time_str_action)
            if i >= len(system_responses):
                return False, 0, 0
            response_match = re.search(r'is:\s*(\d+)', system_responses[i].strip())
            if response_match:
                expected_frame_in_next_action = int(response_match.group(1))
                continue
            return False, 0, 0
        return False, 0, 0
    if expected_frame_in_next_action is not None:
        return False, 0, 0
    return True, tool_call_count, image_add_count

def run_single_qa(cfg: Config) -> Dict[str, Any]:
    device = torch.device(cfg.device_str if cfg.device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, processor = load_model_and_processor(cfg.model_path, device)

    video_id = os.path.splitext(os.path.basename(cfg.video_path))[0]
    fps, frame_count = get_video_metadata(cfg.video_path)
    if frame_count == 0 or fps == 0:
        return {"status": "format_error", "reason": "invalid video metadata"}

    duration_in_seconds = frame_count / fps
    if duration_in_seconds > cfg.long_video_threshold_s:
        n_sample = cfg.num_frames_to_sample_long
        max_w, max_h = cfg.max_frame_width_long, cfg.max_frame_height_long
    else:
        n_sample = cfg.num_frames_to_sample
        max_w, max_h = cfg.max_frame_width, cfg.max_frame_height

    base_frame_dir = cfg.work_dir or tempfile.mkdtemp(prefix="single_qa_frames_")
    os.makedirs(base_frame_dir, exist_ok=True)

    try:
        for attempt in range(cfg.max_retries):
            system_prompt = (
                "You are an expert AI assistant that answers questions about a video by iteratively analyzing it.\n"
                "Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.\n"
                f"Possible actions are:\n1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. The number of frames is fixed, currently {n_sample}.\n"
                "2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30).\n"
                "3. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident."
            )
            if attempt > 1:
                system_prompt = (
                    "You are an expert AI assistant that answers questions about a video.\n"
                    "Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag. "
                    "Your only action is: `output answer: OPTION`: Provide the final answer (e.g., A, B, C...)."
                )

            # Clean per-attempt directory
            if os.path.exists(base_frame_dir):
                shutil.rmtree(base_frame_dir)
            os.makedirs(base_frame_dir, exist_ok=True)

            # Initial uniform sampling
            initial_indices = np.linspace(0, frame_count - 1, n_sample, dtype=int).tolist()
            initial_frames_dir = os.path.join(base_frame_dir, f"{video_id}")
            initial_image_paths = extract_frames(cfg.video_path, initial_indices, initial_frames_dir, max_w, max_h)
            if not initial_image_paths:
                print(f"Warning: Initial frame extraction failed for {video_id}. Retrying...")
                continue

            initial_user_content = [{"type": "text", "text": cfg.question}]
            for path, index in zip(initial_image_paths, initial_indices):
                initial_user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])

            conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user_content},
            ]

            final_answer = None
            for i in range(cfg.max_iterations):
                model_response_str = run_inference(model, processor, conversation_history, device)
                conversation_history.append({"role": "assistant", "content": model_response_str})
                _, action = parse_model_response(model_response_str)
                if action is None:
                    break

                if action.startswith("output answer:"):
                    final_answer = action.replace("output answer:", "").strip()
                    break

                feedback = handle_get_frame_number(action, fps) if "get frame" in action else \
                    handle_choose_frames(action, cfg.video_path, video_id, i + 1, frame_count, base_frame_dir,
                                         n_sample, max_w, max_h)
                if feedback is None:
                    break
                conversation_history.append({"role": "user", "content": feedback})

            if final_answer is not None:
                full_str = stringify_conversation(conversation_history, n_sample)
                is_valid, tool_calls, image_adds = validate_reasoning_process(full_str, n_sample)
                if is_valid:
                    images_used = n_sample * (1 + image_adds)
                    status = None
                    if cfg.correct_answer is not None:
                        status = "correct" if final_answer == cfg.correct_answer else "wrong_answer"
                    return {
                        "status": status or "answered",
                        "final_answer": final_answer,
                        "tool_calls": tool_calls,
                        "images_used": images_used,
                        "attempt": attempt + 1
                    }

        return {"status": "format_error", "reason": "no valid answer after retries"}
    finally:
        if cfg.work_dir is None:
            shutil.rmtree(base_frame_dir, ignore_errors=True)

def main():
    # Set your configuration here.
    CONFIG = Config(
        model_path="path/to/qwen2.5-vl",
        video_path="path/to/video.mp4",
        question="Your question here",
        correct_answer=None,          # e.g., "A" | "B" | "C" | ... or None
        work_dir=None,                # set to a directory to keep frames; None uses a temp dir
        device_str=None,              # e.g., "cuda:0" or "cpu"; None auto-selects
        max_iterations=5,
        max_retries=3,
        num_frames_to_sample=8,
        num_frames_to_sample_long=12,
        max_frame_width=640,
        max_frame_height=360,
        max_frame_width_long=448,
        max_frame_height_long=252,
        long_video_threshold_s=300,
    )

    result = run_single_qa(CONFIG)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()