modified based on `recipe/deepeyes`

entry point is `recipe/scanqa_multiturn/run_deepeyes_grpo.sh`

**Steps:**
1. download and start docker image `verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2`
2. `cd PROJECT_ROOT & pip install -e .`
3. set up wandb
4. download HF dataset from `GumJump/scanqa_images_64_336x224_672x448_multiturn`
5. unzip the files under `DATASET_ROOT/images`, adjust image paths in the train/test.parquet accordingly
6. entry point is `recipe/scanqa_multiturn/run_deepeyes_grpo.sh`

**Note:**
The only files that are relevant to this experiment is 
1. `verl/tools/single_image_resize_tool.py`
2. `recipe/scanqa_multiturn/*`

There are several unrelated folders in this commit:
* folders other than `scanqa_multiturn` under `recipe/`,
* folders other than `examples/data_preprocess/vqa_scanqa_images` under `examples/`