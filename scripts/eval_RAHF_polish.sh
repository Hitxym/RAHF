for BASEMODEL in "gpt-4o" "gpt-3.5-turbo"
do
    for DATASET in "xsum" "writing" "pubmed"
    do
    python dpr/run_RAHF.py \
        --eval_only \
        --from_pretrained=./ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1 \
        --cache_dir=./models \
        --output_file=./results/main/polish/${BASEMODEL} \
        --eval_dataset=./data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json
    done
done

for BASEMODEL in "Qwen2-7B" "Llama-3-8B" "Mistral-7B" "Deepseek-7b"
do
    for DATASET in "xsum" "writing" "squad"
    do
    python dpr/run_RAHF.py \
        --eval_only \
        --from_pretrained=./ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1 \
        --cache_dir=./models \
        --output_file=./results/main/polish/${BASEMODEL} \
        --eval_dataset=./data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json
    done
done