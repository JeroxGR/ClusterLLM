epoch=15
dataset=banking77

# ===== LLM triplets =====
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir checkpoints/finetune-llm/instructor-large-${dataset}-epoch=${epoch} \
    --train_file converted_triplet_results/gpt-oss-120b-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir


# ===== self triplets =====
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir checkpoints/finetune-self/instructor-large-${dataset}-epoch=${epoch} \
    --train_file converted_triplet_results/gpt-oss-120b-self-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir