scale=small
dataset=banking77

python convert_triplet.py \
    --dataset $dataset \
    --pred_path ../1_predict_triplet/predicted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=120_d=67.0_sf_choice_seed=100-openai/gpt-oss-120b-pred.json \
    --output_path converted_triplet_results \
    --data_path ../../datasets/${dataset}/${scale}.jsonl