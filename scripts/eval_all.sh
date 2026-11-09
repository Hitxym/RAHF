# Polish
for BASEMODEL in "gpt-4o" "gpt-3.5-turbo" ;
do
    for DATASET in "xsum" "writing" "pubmed" ;
    do
    python dpr/evaluation.py \
        --eval_dataset="data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json"
    done
done

for BASEMODEL in "Qwen2-7B" "Llama-3-8B" "Mistral-7B" "Deepseek-7b";
do
    for DATASET in "xsum" "writing" "squad" ;
    do
    python dpr/evaluation.py \
        --eval_dataset="data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json"
    done
done

# Rewrite
for BASEMODEL in "gpt-4o" "gpt-3.5-turbo";
do
    for DATASET in "xsum" "writing" "pubmed";
    do
    python dpr/evaluation.py \
        --eval_dataset="data/rewrite/${BASEMODEL}/${DATASET}_rewrite_${BASEMODEL}.raw_data.json"
    done
done

for BASEMODEL in "Qwen2-7B" "Llama-3-8B" "Mistral-7B" "Deepseek-7b";
do
    for DATASET in "xsum" "writing" "squad";
    do
    python dpr/evaluation.py \
        --eval_dataset="data/rewrite/${BASEMODEL}/${DATASET}_rewrite_${BASEMODEL}.raw_data.json"
    done
done

# Generation
for BASEMODEL in "gpt-3.5-turbo" "gpt-4o" "Qwen2-7B" "Llama-3-8B" "Mistral-7B" "Deepseek-7b";
do
    for DATASET in "xsum";
    do
    python dpr/evaluation.py \
        --eval_dataset="data/generation/${BASEMODEL}/${DATASET}_generation_${BASEMODEL}.raw_data.json"
    done
done


# Expand
for BASEMODEL in "gpt-4o" "gpt-3.5-turbo" "Qwen2-7B" "Llama-3-8B" "Mistral-7B" "Deepseek-7b";
do
    for DATASET in "xsum";
    do
    python dpr/evaluation.py \
        --eval_dataset="data/expand/${BASEMODEL}/${DATASET}_expand_${BASEMODEL}.raw_data.json"
    done
done