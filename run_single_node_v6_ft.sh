module load conda

conda activate sam3

torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_v6 \
    --input-dir /pscratch/sd/x/xchong/sam3_finetune/seg_annotation_pipeline2/data4/images \
    --output-dir ./results/data4_v6_ckptv6_p640o50c1 \
    --patch-size 640 \
    --batch-size 1 \
    --confidence 0.1 \
    --overlap-ratio 0.5\
    --prompts "Cortex" "Phloem Fibers" "Phloem" "Hydrated Xylem vessels" "Air-based Pith cells" "Water-based Pith cells" "Dehydrated Xylem vessels" \
    --bpe-path "/pscratch/sd/x/xchong/sam3_finetune/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
    --finetuned-checkpoint "/pscratch/sd/x/xchong/sam3_finetune/sam3/sam3_logs_plants_with_seg_patches_cleaned_640_25_42tiff_7classes_20/checkpoints/checkpoint.pt" \
    --original-checkpoint "/pscratch/sd/x/xchong/sam3_finetune/sam3/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"