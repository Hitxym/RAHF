# Rewrite
for MODELNAME in Qwen2-7B Mistral-7B Llama-3-8B Deepseek-7b
do
    for DATASET in xsum writing squad
    do
        python tools/data_builder.py \
            --model_name=$MODELNAME \
            --dataset=$DATASET \
            --dataset_file=./data/$DATASET \
            --result_dir=./data/rewrite/$MODELNAME \
            --task=rewrite
    done
done

# Polish
for MODELNAME in Qwen2-7B Llama-3-8B Deepseek-7b Mistral-7B    
do
    for DATASET in xsum writing squad
    do
        python tools/data_builder.py \
            --model_name=$MODELNAME \
            --dataset=$DATASET \
            --dataset_file=./data/$DATASET \
            --result_dir=./data/polish/$MODELNAME \
            --task=polish
    done
done

# Expand
for MODELNAME in Qwen2-7B Mistral-7B Llama-3-8B Deepseek-7b 
do
    for DATASET in xsum
    do
        python tools/data_builder.py \
            --model_name=$MODELNAME \
            --dataset=$DATASET \
            --dataset_file=./data/$DATASET \
            --result_dir=./data/expand/$MODELNAME \
            --task=expand
    done
done

# Generation
for MODELNAME in Qwen2-7B Mistral-7B Llama-3-8B Deepseek-7b     
do
    for DATASET in xsum
    do
        python tools/data_builder.py \
            --model_name=$MODELNAME \
            --dataset=$DATASET \
            --dataset_file=./data/$DATASET \
            --result_dir=./data/generation/$MODELNAME \
            --task=generation
    done
done