from dataclasses import dataclass
from typing import Optional, Any, List, Dict
import os
import json

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

from load_data import load_hf_parquet, filter_by_key_frames
from run import run_one_example
from extract_think import extract_last_think


@dataclass
class Config:
    # Data
    dataset_path: str = "/workspace/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet"
    hf_token: Optional[str] = None
    hf_revision: Optional[str] = None

    # Model
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 512
    do_sample: bool = True

    # Output
    output_jsonl: str = "/home/geng/git/verl_0.6/outputs/reasoning.jsonl"

    # Utility
    limit_rows: Optional[int] = None  # set to an int for quick tests


def _normalize_images(images: List[Any]) -> List[Any]:
    # Accept strings (paths) or dicts like {"image": "/path"} as-is; run_one_example can handle both.
    return images


def main(cfg: Config) -> None:
    # Load dataset
    ds = load_hf_parquet(
        cfg.dataset_path,
        token=cfg.hf_token,
        revision=cfg.hf_revision,
    )

    # Ensure key_intervals exist
    ds = ds.map(filter_by_key_frames)

    # Load model/processor once
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Prepare output
    os.makedirs(os.path.dirname(cfg.output_jsonl), exist_ok=True)
    with open(cfg.output_jsonl, "w") as f:
        pass  # truncate

    # Iterate rows
    n = len(ds)
    for row_idx in tqdm(range(n)):
        if cfg.limit_rows is not None and row_idx >= cfg.limit_rows:
            break

        row: Dict[str, Any] = ds[row_idx]
        images = _normalize_images(row["images"])
        question = row["extra_info"]["question"]
        intervals = row.get("key_intervals", [])
        question_id = row["extra_info"]["question_id"]

        if not intervals:
            continue

        # Run for each interval independently
        for interval in intervals:
            text = run_one_example(
                images=images,
                groundtruth_intervals=int(interval),
                question=question,
                model=model,
                processor=processor,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
            )

            think = extract_last_think(text)

            record = {
                "question_id": row["extra_info"]["question_id"],
                "row_index": row_idx,
                "interval": int(interval),
                "question": question,
                "output": text,
                "think": think,
            }

            with open(cfg.output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Config())