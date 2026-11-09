# polish task
for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    for BASEMODEL in "gpt-3.5-turbo" "gpt-4o"
    do
        for DATASET in "xsum" "writing" "pubmed"
        do
        python other_models/supervised.py \
            --output_file=./other_model_results/polish/${DATASET}_polish_${BASEMODEL} \
            --dataset_file=./data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL} \
            --model_name=roberta-large-openai-detector \
            --dataset=${DATASET} \
            --cache_dir=./models \
            --seed=42
        done
    done
done

# rewrite task
for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    for BASEMODEL in "gpt-3.5-turbo" "gpt-4o"
    do
        for DATASET in "xsum" "writing" "pubmed"
        do
        python other_models/supervised.py \
            --output_file=./other_model_results/rewrite/${DATASET}_rewrite_${BASEMODEL} \
            --dataset_file=./data/rewrite/${BASEMODEL}/${DATASET}_rewrite_${BASEMODEL} \
            --model_name=${MODEL} \
            --dataset=${DATASET} \
            --cache_dir=./models \
            --seed=42
        done
    done
done

# polish task for open source models
for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    for BASEMODEL in "Qwen2-7B" "Deepseek-7b" "Llama-3-8B" "Mistral-7B"
    do
        for DATASET in "xsum" "writing" "squad"
        do
        python other_models/supervised.py \
            --output_file=./other_model_results/polish/${DATASET}_polish_${BASEMODEL} \
            --dataset_file=./data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL} \
            --model_name=${MODEL} \
            --dataset=${DATASET} \
            --cache_dir=./models \
            --seed=42
        done
    done
done

# rewrite task for open source models
for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    for BASEMODEL in "Qwen2-7B" "Deepseek-7b" "Llama-3-8B" "Mistral-7B"
    do
        for DATASET in "xsum" "writing" "squad"
        do
        python other_models/supervised.py \
            --output_file=./other_model_results/rewrite/${DATASET}_rewrite_${BASEMODEL} \
            --dataset_file=./data/rewrite/${BASEMODEL}/${DATASET}_rewrite_${BASEMODEL} \
            --model_name=${MODEL} \
            --dataset=${DATASET} \
            --cache_dir=./models \
            --seed=42
        done
    done
done

