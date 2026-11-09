# SFT
for BASEMODEL in "gpt-3.5-turbo" "gpt-4o"
do
    for DATANUM in "1000" "2000" "3000"
    do
        for DATASET in "xsum" "writing" "pubmed"
        do
        python dpr/run_RAHF.py \
            --eval_only \
            --base_model=finetuned_gpt_neo_2.7B_${DATANUM}_sft \
            --cache_dir=./ckpt \
            --output_file=./results/ablation/${BASEMODEL}/ablation_exp_${DATANUM} \
            --eval_dataset="data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json" \
            --SPOtrained=False
        done
    done
done

# ORPO
for BASEMODEL in "gpt-3.5-turbo" "gpt-4o"
do
    for DATASET in "xsum" "writing" "pubmed"
    do
        python dpr/run_RAHF.py \
            --eval_only \
            --base_model=ai_detection_500_orpo \
            --cache_dir=./ckpt \
            --output_file=./results/ablation/${BASEMODEL}/orpo \
            --eval_dataset="data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json" \
            --SPOtrained=False
    done
done

# RLHF
for BASEMODEL in "gpt-3.5-turbo" "gpt-4o"
do
    for DATASET in "xsum" "writing" "pubmed"
    do
        python dpr/run_RAHF.py \
            --eval_only \
            --base_model=ai_detection_500_rlhf \
            --cache_dir=./ckpt \
            --output_file=./results/ablation/${BASEMODEL}/rlhf \
            --eval_dataset="data/polish/${BASEMODEL}/${DATASET}_polish_${BASEMODEL}.raw_data.json" \
            --SPOtrained=False
    done
done
