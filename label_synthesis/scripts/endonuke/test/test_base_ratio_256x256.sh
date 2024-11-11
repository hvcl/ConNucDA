device=$1
# test_captions=$2
# test_captions="high"
# test_captions: "high nucleus area ratio" "medium nucleus area ratio" "low nucleus area ratio"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

# Low
python test_class.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/endonuke/joint/base_256x256/240213_061059_ratio-256x256/checkpoint.300000.pt \
    --sample_timesteps 200 \
    --start_sample_idx 67 \
    --num_samples=67 \
    --test_batch_size=1 \
    --dataset endonuke \
    --num_classes 4 \
    --save_path=results/endonuke/base_256x256/240213_061059_ratio-256x256/300000_200_2/base.png \
    --save_img_path=outputs/endonuke/base_256x256/240213_061059_ratio-256x256/300000_200_2/ \
    --caption_list_dir data/eval_samples/endonuke/endonuke_test_ratio.json

# Medium

# High
