export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
# export CUDA_VISIABLE_DEVICES=4,5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# --caption_list_dir data/caption_files/lizard+density \
# --caption_list_dir data/caption_files/lizard+density-ratio \
# --caption_list_dir data/caption_files/lizard+density-ratio-class \

accelerate launch --config_file scripts/lizard/train_base_ratio_class_256x256.yaml \
    train.py \
    --root_dir /Dataset/lizard/NASDM/lizard_split_norm_256 \
    --caption_list_dir data/caption_files/lizard+ratio \
    --test_caption_files data/eval_samples/lizard/lizard_test.jsonl \
    --dataset lizard \
    --num_classes 7 \
    --exp_name ratio-class_256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 6 \
    --batch_size 6 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type lizard \
    --split train \
    --fp16 ${@:1} \
    --num_workers 0 \
    --no_condition_on_text

# --noise_schedules linear linear --noise_schedules_lbl linear linear \