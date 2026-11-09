for LANGUAGE in "Spanish" "Portuguese" "Chinese";
do
    python dpr/run_RAHF.py \
        --datanum=500 \
        --task_name=${LANGUAGE}_polish_500 \
        --epochs=2 \
        --lr=1e-4 \
        --beta=0.05 \
        --train_dataset=./data/multilang/data/${LANGUAGE}_train_500_polish_gpt-3.5-turbo.raw_data.json \
        --eval_dataset=./data/multilang/data/${LANGUAGE}_test_150_polish_gpt-3.5-turbo.raw_data.json
done