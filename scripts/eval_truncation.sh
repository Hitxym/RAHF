BASEMODELS=("gpt-3.5-turbo" "gpt-4o")
LENS=("30" "60" "90" "120" "150" "180")

# SPO
for BASEMODEL in "${BASEMODELS[@]}"
do
    for LEN in "${LENS[@]}"
    do
    python dpr/run_RAHF.py \
        --eval_only \
        --from_pretrained=./ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1 \
        --cache_dir=./models \
        --output_file=./other_model_results/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
        --eval_dataset=./data/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN}.raw_data.json
    done
done

# fast_detect_gpt_gptj_neo2.7
for BASEMODEL in "${BASEMODELS[@]}"
do
        for LEN in "${LENS[@]}"
        do
                python ./other_models/fast_detect_gpt.py \
                --dataset_file=./data/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --output_file=./other_model_results/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --scoring_model_name=gpt-neo-2.7B \
                --reference_model_name=gpt-j-6B \
                --cache_dir=./models \
                --seed=42 \
                --dataset=xsum
        done
done

# RoB
for BASEMODEL in "${BASEMODELS[@]}"
do
    for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
    do
        for LEN in "${LENS[@]}"
        do
            python other_models/supervised.py \
                --output_file=./other_model_results/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --dataset_file=./data/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --model_name=${MODEL} \
                --dataset=${DATASET} \
                --cache_dir=./models \
                --seed=42
        done
    done
done

# Likelihood
for BASEMODEL in "${BASEMODELS[@]}"
do
        for LEN in "${LENS[@]}"
        do
                python ./other_models/baselines.py \
                --dataset_file=./data/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --output_file=./other_model_results/truncation/xsum_polish_${BASEMODEL}_truncation_${LEN} \
                --scoring_model_name=gpt-neo-2.7B \
                --cache_dir=./models \
                --seed=42 \
                --dataset=xsum
        done
done