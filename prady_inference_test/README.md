# SINGLE NODE(4 GPUS)

### Performance (Perlmutter A100)

| Setup        | Slices | Time (min) |Per Slice (s) |
|--------------|--------|------------|--------------|
| **4 GPUs**   | 2160   | **6.1 min**| **0.17 s**   |

## Hardware

- NVIDIA A100 GPUs
- CUDA 12.4
- torchrun + NCCL

## Files

- run_sam3_4gpu.py   : 4-GPU inference script
- results_viz.ipynb : Visualization notebook
- predicted_mask.png : Example output

## Checkpoints (Not Included)

Requires local checkpoints:

- sam3-002.pt
- checkpoint-001.pt

Update paths inside scripts.

## Run (4 GPUs)

```bash
torchrun --nproc_per_node=4 run_sam3_4gpu.py \
  --input_dir /path/to/images \
  --out_dir /path/to/output \
  --bpe_path /path/to/bpe.txt.gz \
  --sam3_pt /path/to/sam3.pt \
  --finetune_ckpt /path/to/ckpt.pt \
  --confidence 0.5 \
  --amp \
  --prompts "cortex" "Phloem Fibers" "Xylem vessels" "Pith cells" "outer cells"
