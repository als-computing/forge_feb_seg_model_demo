#!/usr/bin/env python3
# ------------------------------------------------------------
# SAM3 4-GPU Distributed Inference (NERSC Compatible)
# ------------------------------------------------------------

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------------
# Make sure local sam3 repo is found
# ------------------------------------------------------------------

#sys.path.append("/pscratch/sd/e/elavarpa/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------

def setup_ddp():

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, world, local_rank


def cleanup_ddp():

    if dist.is_initialized():
        dist.destroy_process_group()


def load_state_dict_flexible(path):

    ckpt = torch.load(path, map_location="cpu")

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # remove module. if present
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    return state


def load_tiff(path):

    img = Image.open(path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def save_mask(mask, out_path):

    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)

    Image.fromarray(mask).save(out_path, compression="tiff_deflate")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--bpe_path", required=True)
    parser.add_argument("--sam3_pt", required=True)
    parser.add_argument("--finetune_ckpt", default=None)

    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--prompts", nargs="+", required=True)

    args = parser.parse_args()


    # ------------------------------------------------------------
    # DDP init
    # ------------------------------------------------------------

    rank, world, local_rank = setup_ddp()

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\nRunning on {world} GPUs\n")


    # ------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------

    input_dir = Path(args.input_dir)

    files = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff"))
    )

    if rank == 0:
        print(f"Total slices: {len(files)}")


    # ------------------------------------------------------------
    # Shard files across GPUs
    # ------------------------------------------------------------

    my_files = files[rank::world]

    print(f"[Rank {rank}] Processing {len(my_files)} slices")


    # ------------------------------------------------------------
    # Build model (LOCAL ONLY)
    # ------------------------------------------------------------

    if rank == 0:
        print("Loading model...")


    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        checkpoint_path=args.sam3_pt,     # <-- disables HF
        load_from_HF=False               # <-- disables HF
    )


    # Load finetune if given
    if args.finetune_ckpt:

        ft = load_state_dict_flexible(args.finetune_ckpt)

        model.load_state_dict(ft, strict=False)

        if rank == 0:
            print("Loaded finetune checkpoint")


    model.eval().to(device)


    # ------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------

    processor = Sam3Processor(
        model,
        confidence_threshold=args.confidence
    )


    # ------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------

    out_root = Path(args.out_dir)

    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)

        for p in args.prompts:
            (out_root / p.replace(" ", "_")).mkdir(exist_ok=True)


    dist.barrier()


    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    if rank == 0:
        print("\nStarting inference...\n")


    start = time.time()


    autocast = torch.cuda.amp.autocast if args.amp else torch.no_grad


    for path in tqdm(my_files, disable=(rank != 0)):

        img = load_tiff(path)

        with torch.no_grad():

            state = processor.set_image(img)

            processor.reset_all_prompts(state)


            for prompt in args.prompts:

                with autocast():

                    state = processor.set_text_prompt(
                        state=state,
                        prompt=prompt
                    )

                masks = state.get("masks", None)

                if masks is None:
                    continue


                # Merge multi-mask → single mask
                # m = masks.max(dim=0)[0]
                if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
                    # No detections → return empty mask
                    H, W = img.size[1], img.size[0]
                    m = torch.zeros((H, W), device=device, dtype=torch.float32)
                else:
                    m = masks.max(dim=0)[0] 
                
                m = m.squeeze().cpu().numpy()


                out_dir = out_root / prompt.replace(" ", "_")

                out_path = out_dir / f"{path.stem}_mask.tif"

                save_mask(m, out_path)


    torch.cuda.synchronize()

    end = time.time()


    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    local_time = torch.tensor([end - start], device=device)

    dist.all_reduce(local_time, op=dist.ReduceOp.MAX)


    if rank == 0:

        total = local_time.item()

        print("\n✓ Finished")
        print(f"Total time : {total:.1f} sec")
        print(f"Minutes    : {total/60:.2f}")
        print(f"Slices     : {len(files)}")
        print(f"Per slice  : {total/len(files):.3f} sec")


    cleanup_ddp()


# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
