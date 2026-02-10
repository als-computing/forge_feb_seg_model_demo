module load conda

conda activate sam3

torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_v4 \
    --input-dir ./sample_data/rec20211222_125057_petiole4 \
    --output-dir ./rec20211222_125057_petiole4_v4_ckptv2 \
    --patch-size 640 \
    --batch-size 1 \
    --confidence 0.5 \
    --prompts "Cortex" "Phloem Fibers" "Air-based Pith cells" "Water-based Pith cells" "Xylem vessels" \
    --bpe-path "/pscratch/sd/x/xchong/sam3_finetune/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
    --finetuned-checkpoint "/pscratch/sd/x/xchong/sam3_finetune/sam3/sam3_logs_plants_with_seg_patches_cleaned_640_25_20tiff_20_corr/checkpoints/checkpoint.pt" \
    --original-checkpoint "/pscratch/sd/x/xchong/sam3_finetune/sam3/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"