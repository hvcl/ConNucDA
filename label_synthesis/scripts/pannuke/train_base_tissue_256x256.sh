export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_VISIABLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# --caption_list_dir data/caption_files/pannuke+density \
# --caption_list_dir data/caption_files/pannuke+density-ratio \
# --caption_list_dir data/caption_files/pannuke+density-ratio-class \

accelerate launch --config_file scripts/pannuke/train_base_tissue_256x256.yaml \
    train.py \
    --root_dir /Dataset/pannuke/splits \
    --caption_list_dir data/caption_files/pannuke+tissue \
    --test_caption_files data/eval_samples/pannuke/pannuke_test_tissue.jsonl \
    --dataset pannuke \
    --num_classes 6 \
    --exp_name tissue-256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 300000 \
    --save_every 10000 \
    --max_batch_size 6 \
    --batch_size 6 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type pannuke \
    --split train \
    --fp16 ${@:1} \
    --num_workers 0 \
    --no_condition_on_text

# --noise_schedules linear linear --noise_schedules_lbl linear linear \
