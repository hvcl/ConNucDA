device=$1
# test_captions=$2
# test_captions="high"
# test_captions: "high nucleus area ratio" "medium nucleus area ratio" "low nucleus area ratio"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

# Low
python test_class.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/lizard/joint/base_256x256/240214_165437_ratio-256x256/checkpoint.300000.pt \
    --sample_timesteps 100 \
    --start_sample_idx 390 \
    --num_samples=395 \
    --test_batch_size=1 \
    --dataset lizard \
    --num_classes 7 \
    --save_path=results/lizard/base_256x256/240214_165437_ratio-256x256/300000/base.png \
    --save_img_path=outputs/lizard/base_256x256/240214_165437_ratio-256x256/300000/ \
    --caption_list_dir data/eval_samples/lizard/lizard_test_ratio-class.json

# Medium

# High
