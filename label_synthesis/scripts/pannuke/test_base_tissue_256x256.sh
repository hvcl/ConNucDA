device=$1
# test_captions=$2
# test_captions="high"
# test_captions: "high nucleus area ratio" "medium nucleus area ratio" "low nucleus area ratio"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

# Low
python test_class.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/pannuke/joint/base_256x256/240305_014319_tissue-256x256/checkpoint.260000.pt \
    --sample_timesteps 100 \
    --start_sample_idx 580 \
    --num_samples=20 \
    --test_batch_size=1 \
    --dataset pannuke \
    --num_classes 6 \
    --save_path=results/pannuke/base_256x256/240305_014319_tissue-256x256/260000/base.png \
    --save_img_path=outputs/pannuke/base_256x256/240305_014319_tissue-256x256/260000/ \
    --caption_list_dir data/eval_samples/pannuke/pannuke_test_tissue.json

# Medium

# High