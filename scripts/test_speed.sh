# SPO
python dpo/run_ours_dpo.py \
        --eval_only \
        --from_pretrained=./ckpt/ai_detection_polish_500_dpo_lr_0.0001_beta_0.05_a_1 \
        --output_file=./results/speed/xsum_polish_gpt3.5_10 \
        --eval_dataset=./data/truncation/xsum_polish_gpt-3.5-turbo_10.raw_data.json \
        --cache_dir=./models \
        --seed=42

# Entropy, Likelihood LogRank
python ./other_models/baselines.py \
    --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10 \
    --output_file=./results/speed/xsum_polish_gpt3.5_10 \
    --scoring_model_name=gpt-neo-2.7B \
    --cache_dir=./models \
    --seed=42 \
    --dataset=xsum


# detect_gpt
python ./other_models/detect_gpt.py \
    --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10 \
    --output_file=./results/speed/xsum_polish_gpt3.5_10 \
    --scoring_model_name=gpt-neo-2.7B \
    --mask_filling_model_name=t5-3b \
    --cache_dir=./models \
    --seed=42 \
    --dataset=xsum


# LRR NPR
python ./other_models/detect_llm.py \
    --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10.t5-3b.perturbation_100 \
    --output_file=./results/speed/xsum_polish_gpt3.5_10 \
    --scoring_model_name=gpt-neo-2.7B \
    --cache_dir=./models \
    --seed=42 \
    --dataset=xsum


# fast_detect_gpt_gptj_neo2.7
python ./other_models/fast_detect_gpt.py \
    --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10 \
    --output_file=./results/speed/xsum_polish_gpt3.5_10 \
    --scoring_model_name=gpt-neo-2.7B \
    --reference_model_name=gpt-j-6B \
    --cache_dir=./models \
    --seed=42 \
    --dataset=xsum

# dnagpt
python ./other_models/dna_gpt.py \
    --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10 \
    --output_file=./results/speed/xsum_polish_gpt3.5_10 \
    --base_model_name=gpt-neo-2.7B \
    --cache_dir=./models \
    --seed=42 \
    --dataset=xsum

# roberta-base and large
for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    python other_models/supervised.py \
        --dataset_file=./data/truncation/xsum_polish_gpt-3.5-turbo_10 \
        --output_file=./results/speed/xsum_polish_gpt3.5_10 \
        --model_name=${MODEL} \
        --dataset=xsum \
        --cache_dir=./models \
        --seed=42
done