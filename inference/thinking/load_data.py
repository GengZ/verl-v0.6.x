from typing import List, Optional, Union
from datasets import load_dataset, Dataset
import os

def load_hf_parquet(
    source: Union[str, List[str]],
    *,
    split: Optional[str] = None,
    columns: Optional[List[str]] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dataset:
    """
    Load a Hugging Face Parquet dataset fully into memory (no streaming).

    Parameters
    ----------
    source :
        - Local parquet path or glob (e.g. "data/*.parquet"),
        - HTTP(S) URL to a parquet file on the Hub,
        - OR a Hub dataset repo id (e.g. "org/dataset").
    split : str, optional
        Required when `source` is a Hub dataset repo id.
    columns : list[str], optional
        If provided, keep only these columns in memory.
    token : str, optional
        HF token for private datasets/files.
    revision : str, optional
        Git revision (branch/tag/commit) for Hub datasets or URLs.

    Returns
    -------
    datasets.Dataset
        Fully loaded Dataset object (in-memory).

    Examples
    --------
    # 1) Load local parquet(s)
    ds = load_hf_parquet("data/*.parquet")

    # 2) Load from Hub repo
    ds = load_hf_parquet("openai/ai2d", split="train")

    # 3) Load from a specific parquet file URL
    url = "https://huggingface.co/datasets/.../resolve/main/train-00000-of-00005.parquet"
    ds = load_hf_parquet(url, columns=["text", "label"])
    """
    if split is not None and not _looks_like_parquet_path_or_url(source):
        # Hub dataset
        ds = load_dataset(source, split=split, token=token, revision=revision)
    else:
        # Local parquet(s) or direct URL
        ds = load_dataset(
            "parquet",
            data_files=source,
            split="train",
            token=token,
            revision=revision,
        )

    if columns is not None:
        ds = ds.remove_columns([c for c in ds.column_names if c not in columns])

    return ds


def _looks_like_parquet_path_or_url(src: Union[str, List[str]]) -> bool:
    def _one(s: str) -> bool:
        s_lower = s.lower()
        return (
            s_lower.endswith(".parquet")
            or s_lower.startswith("http://")
            or s_lower.startswith("https://")
            or any(ch in s for ch in ("*", "/", os.sep))
        )
    if isinstance(src, list):
        return all(_one(s) for s in src)
    return _one(src)


def filter_by_key_frames(row):
    down_sampled_factor = 9
    keep_frames = []
    keep_intervals = []

    key_frames = row['extra_info']['key_frame_indexes']
    key_frames = [int(f) for f in key_frames]

    # only preserve the key frames if divided by down sampled factor 
    for frame in key_frames:
        if frame % down_sampled_factor == 0:
            keep_frames.append(frame)
            keep_intervals.append(frame // down_sampled_factor)
            if frame // down_sampled_factor - 1 not in keep_intervals and frame // down_sampled_factor - 1 >= 0:
                keep_intervals.append(frame // down_sampled_factor - 1)

    row['key_intervals'] = keep_intervals

    return row

if __name__ == "__main__":
    ds = load_hf_parquet("/home/geng/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet")

    print(len(ds))

    ds = ds.map(filter_by_key_frames)

    print(len(ds))
    print(ds[0]['key_intervals'])
    print(ds[0]['extra_info']['key_frame_indexes'])
    print(ds[0]['extra_info']['question'])
    print(ds[0]['images'])
