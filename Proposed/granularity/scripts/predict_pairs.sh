# export OPENAI_API_KEY="OPENAI_API_KEY"
# for dataset in banking77
# do
#     link_path=sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json
#     OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_pairs.py \
#         --dataset $dataset \
#         --data_path $link_path \
#         --model_name gpt-4-0314 \
#         --openai_org "OPENAI_ORG" \
#         --prompt_file prompts_pair_exps_pair_v3.json \
#         --temperature 0
# done


export OPENAI_API_KEY="YOUR_GROQ_API_KEY_HERE"
export OPENAI_API_BASE="https://api.groq.com/openai/v1"

for dataset in banking77
do
    link_path=sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_pairs.py \
        --dataset $dataset \
        --data_path $link_path \
        --model_name openai/gpt-oss-120b \
        --openai_org "none" \
        --prompt_file prompts_pair_exps_pair_v3.json \
        --temperature 0 \
        --delay 2
done