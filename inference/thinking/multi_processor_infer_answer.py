import json
import os
import multiprocessing as mp
from dataclasses import dataclass, field

from PIL import Image
from typing import Iterable, Any, List, Dict, Union, Sequence, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq

from load_data import load_hf_parquet
from extract_think import extract_last_summary

from infer_one_answer import run_one_example, _to_pil_rgb, load_jsonl

@dataclass
class Config:
    # Data
    dataset_path: str = "/workspace/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet"
    input_jsonl: str = "/home/geng/git/verl_0.6/outputs/reasoning.jsonl"
    output_jsonl: str = "/home/geng/git/verl_0.6/outputs/answer.jsonl"
    hf_token: Optional[str] = None
    hf_revision: Optional[str] = None

    # Model
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 512
    do_sample: bool = True

    # Parallel
    gpus: Optional[List[int]] = field(default_factory=lambda: [0, 1])  # e.g., [0,1,2]; defaults to CUDA_VISIBLE_DEVICES or all available
    procs_per_gpu: int = 2            # processes per GPU

    # Utility
    limit_items: Optional[int] = None  # limit input JSONL items for quick tests


def _discover_gpu_ids(gpus: Optional[List[int]]) -> List[int]:
    if gpus:
        return list(map(int, gpus))
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        vals = [s.strip() for s in env.split(",") if s.strip() != ""]
        try:
            return [int(v) for v in vals]
        except ValueError:
            # If CUDA_VISIBLE_DEVICES is remapped like "0,1", still treat as indices
            return list(range(len(vals)))
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []  # CPU fallback


def _build_parquet_lookup(ds: Dataset) -> Dict[str, Any]:
    # Map question_id -> (row_index, row)
    lut: Dict[str, Any] = {}
    for idx in range(len(ds)):
        row = ds[idx]
        qid = row["extra_info"]["question_id"]
        lut[qid] = (idx, row)
    return lut


def _worker(rank: int, gpu_id: Optional[int], tasks: List[Dict[str, Any]], cfg: Config) -> None:
    # Isolate device visibility for this worker
    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load model/processor
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load dataset and lookup
    ds = load_hf_parquet(cfg.dataset_path, token=cfg.hf_token, revision=cfg.hf_revision)
    lut = _build_parquet_lookup(ds)

    shard_path = f"{cfg.output_jsonl}.rank{rank}.jsonl"
    os.makedirs(os.path.dirname(cfg.output_jsonl), exist_ok=True)

    for item in tasks:
        try:
            qid = item.get("question_id")
            interval = int(item.get("interval"))
            think = item.get("think", "")

            if qid not in lut:
                continue
            row_index, row = lut[qid]

            # Prepare frames
            initial_frames = [_to_pil_rgb(im) for im in row["images"]]
            interval_frames = [_to_pil_rgb(im) for im in row["non_key_frame_image_paths"][str(interval)]]

            # Question/Answer
            question = row["extra_info"]["question"]
            answer = row["extra_info"]["answer"]

            # Generate
            output_text = run_one_example(
                initial_frames=initial_frames,
                interval_frames=interval_frames,
                i=interval,
                question=question,
                answer=answer,
                selection_thinking=think,
                model=model,
                processor=processor,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
            )

            record = {
                "question_id": qid,
                "row_index": row_index,
                "interval": interval,
                "question": question,
                "answer": answer,
                "thinking": think,
                "output": output_text,
                "summary": extract_last_summary(output_text),
            }

            with open(shard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            # Skip failed item but continue processing others
            err_rec = {
                "error": str(e),
                "question_id": item.get("question_id"),
                "interval": item.get("interval"),
                "rank": rank,
            }
            with open(shard_path + ".errors", "a", encoding="utf-8") as f:
                f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")


def multi_process_reasoning_answer(cfg: Config) -> None:
    # Load input tasks from JSONL
    items = load_jsonl(cfg.input_jsonl)
    if cfg.limit_items is not None:
        items = items[: cfg.limit_items]

    # Determine workers
    gpu_ids = _discover_gpu_ids(cfg.gpus)
    if not gpu_ids:
        total_workers = 1
        assigned = [(0, None)]
    else:
        total_workers = max(1, len(gpu_ids) * max(1, cfg.procs_per_gpu))
        assigned = [(rank, gpu_ids[rank % len(gpu_ids)]) for rank in range(total_workers)]

    # Shard tasks
    shards: Dict[int, List[Dict[str, Any]]] = {rank: [] for rank, _ in assigned}
    for idx, item in enumerate(items):
        rank = idx % total_workers
        shards[rank].append(item)

    # Ensure output dir exists and clear final file
    os.makedirs(os.path.dirname(cfg.output_jsonl), exist_ok=True)
    if os.path.exists(cfg.output_jsonl):
        os.remove(cfg.output_jsonl)

    # Spawn workers
    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []
    for rank, gpu_id in assigned:
        if not shards[rank]:
            continue
        p = ctx.Process(target=_worker, args=(rank, gpu_id, shards[rank], cfg))
        p.start()
        procs.append(p)

    # Join
    for p in procs:
        p.join()

    # Merge shards
    with open(cfg.output_jsonl, "w", encoding="utf-8") as out_f:
        for rank, _ in assigned:
            shard_path = f"{cfg.output_jsonl}.rank{rank}.jsonl"
            if os.path.exists(shard_path):
                with open(shard_path, "r", encoding="utf-8") as sf:
                    for line in sf:
                        out_f.write(line)
                os.remove(shard_path)

    # Optionally keep error shards; do not merge them automatically


def main(cfg: Config) -> None:
    multi_process_reasoning_answer(cfg)


if __name__ == "__main__":
    main(Config())