# export OPENAI_API_KEY="YOUR_GROQ_API_KEY_HERE"
# for dataset in banking77
# do
#     link_path=sampled_triplet_results/${dataset}_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json
#     # link_path=sampled_triplet_results/${dataset}_embed=instructor_s=large_m=1024_d=67.0_sf_choice_seed=100.json
#     OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_triplet.py \
#         --dataset $dataset \
#         --data_path $link_path \
#         --openai_org "OPENAI_ORG" \
#         --model_name openai/gpt-oss-120b \
#         --temperature 0
# done

export OPENAI_API_KEY="YOUR_GROQ_API_KEY_HERE"
export OPENAI_API_BASE="https://api.groq.com/openai/v1"

for dataset in banking77
do
    link_path=sampled_triplet_results/${dataset}_embed=instructor_s=small_m=120_d=67.0_sf_choice_seed=100.json

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_triplet.py \
        --dataset $dataset \
        --data_path $link_path \
        --openai_org "none" \
        --model_name openai/gpt-oss-120b \
        --temperature 0 \
        --delay 2
done