#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

# Import scorer from your repo
from inference.utils import scanqa_aggregate_results

def extract_answer(text: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

def prepare_items(raw_list):
    items = []
    for r in raw_list:
        gt_answers = r.get("gt_answers", [])
        pred_raw = r.get("pred_response", "")
        pred = extract_answer(pred_raw)
        items.append({
            "pred_response": pred,
            "gt_response": {"answers": gt_answers},
        })
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Path to results JSON file",
    )
    args = parser.parse_args()
    path = Path(args.results)

    with path.open("r") as f:
        data = json.load(f)

    items = prepare_items(data)
    metrics = scanqa_aggregate_results(items)

    # Pretty-print metrics
    print(json.dumps(metrics, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()