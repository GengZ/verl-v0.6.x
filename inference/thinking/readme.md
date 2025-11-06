I'll scan the `inference/thinking` files to understand their roles, then draft a concise README grouped by purpose.

[9 tools called]

### README — `inference/thinking/`

This folder contains a two-stage pipeline for multimodal “thinking” on video keyframes:
- Stage 1: Generate interval-selection reasoning (<think>) for which adjacent keyframes (i → i+1) contain key evidence.
- Stage 2: Verify evidence using retrieved in-between frames and emit a concise evidence summary + final answer.

### How the pieces fit
- Stage 1 produces `outputs/reasoning.jsonl` with per-question interval and `<think>` reasoning.
- Stage 2 consumes `outputs/reasoning.jsonl` + the HF parquet dataset, verifies evidence, and writes results to `outputs/answer.jsonl` (or `outputs/reasoning_answer.jsonl` in the single-process script).

## Files grouped by purpose

### Data I/O and preprocessing
- `load_data.py`
  - `load_hf_parquet(...)`: Loads local/Hub parquet(s) into a `datasets.Dataset`.
  - `filter_by_key_frames(row)`: Derives downsampled key intervals from `extra_info.key_frame_indexes` using a factor of 9 and stores them in `row['key_intervals']`.

### Stage 1 — Interval selection “thinking” (generate <think>)
- `run.py`
  - `run_one_example(images, groundtruth_intervals, question, model, processor, ...)`: Core routine that builds the selection-justification prompt and generates text containing a `<think>...</think>` block.
  - CLI example section loads a Qwen2.5-VL model and runs a single example.
- `loop.py`
  - Orchestrates Stage 1 over an entire parquet dataset:
    - Loads dataset, maps `filter_by_key_frames` to create `key_intervals`.
    - Loads model once and calls `run_one_example` for each `(row, interval)`.
    - Extracts `<think>` via `extract_last_think` and appends records to `outputs/reasoning.jsonl`.
- `main.py`
  - Self-contained demo that hard-codes image paths, question, and interval and runs the same “selection reasoning” idea (uses Qwen2.5-VL-32B by default). Useful for quick sanity checks.

### Stage 2 — Evidence verification + final answer
- `infer_one_answer.py`
  - Builds a verification-style prompt that includes:
    - Initial downsampled frames (Frame-k),
    - Retrieved in-between frames for interval i→i+1 (Between-i-i+1-t),
    - The question, the ground-truth answer, and prior selection “thinking.”
  - Helpers:
    - `load_jsonl(path)`: Read `reasoning.jsonl`.
    - `pair_jsonl_with_parquet(jsonl_data, parquet)`: Attach dataset rows by `question_id`.
    - `run_one_example(...)`: Generates `<evidence_summary>...</evidence_summary>` and `<final_answer>...</final_answer>`.
  - Writes per-item outputs to `outputs/reasoning_answer.jsonl` including a parsed `summary`.
- `multi_processor_infer_answer.py`
  - Parallelized Stage 2 with sharded workers across GPUs/CPUs:
    - `Config`: paths, model id, generation params, and parallelism (GPU IDs, processes per GPU).
    - Spawns workers, each loading the model + dataset once, processes a shard of `reasoning.jsonl`, writes `answer.jsonl.rank{N}.jsonl`, merges into `outputs/answer.jsonl`, and keeps `.errors` logs if any.

### Prompt parsing utilities
- `extract_think.py`
  - `extract_last_think(text)`: Pulls the last `<think>...</think>` block.
  - `extract_last_summary(text)`: Pulls the last `<evidence_summary>...</evidence_summary>` block.

### Examples and scratch
- `example.json`
  - Minimal examples showing the Stage 1 record (with `<think>`) and a Stage 2 record (with final `<evidence_summary>` + `<final_answer>`).
- `tmp.txt`
  - Empty placeholder (scratch).

## Minimal usage

- Stage 1 (produce selection reasoning):
```bash
python /home/geng/git/verl_0.6/inference/thinking/loop.py
# -> writes /home/geng/git/verl_0.6/outputs/reasoning.jsonl
```

- Stage 2 (verify evidence + answer):
  - Single-process:
```bash
python /home/geng/git/verl_0.6/inference/thinking/infer_one_answer.py
# -> writes /home/geng/git/verl_0.6/outputs/reasoning_answer.jsonl
```
  - Multiprocess:
```bash
python /home/geng/git/verl_0.6/inference/thinking/multi_processor_infer_answer.py
# -> writes /home/geng/git/verl_0.6/outputs/answer.jsonl
```

- Quick demo of Stage 1 on a fixed example:
```bash
python /home/geng/git/verl_0.6/inference/thinking/main.py
```

Notes
- Edit dataset/model paths and generation params via the `Config` classes or constants in each script.
- Requires `torch`, `transformers`, `datasets`, `Pillow`, and `tqdm`.