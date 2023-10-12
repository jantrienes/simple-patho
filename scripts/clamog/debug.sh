#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 05:00:00
#SBATCH --partition=GPUampere,GPUhopper

eval "$(conda shell.bash hook)"
conda activate simple-patho

export WANDB_PROJECT=simplepatho-clamog
dataset=d2h-v1-aligned-para
if [[ $MODEL_NAME_OR_PATH == *"models/"* ]]; then
    run_name=$(echo "$MODEL_NAME_OR_PATH" | awk -F'/' '{print $NF}')
else
    run_name=$MODEL_NAME_OR_PATH
fi
run_name=${run_name}-debug
data_dir=${PWD}/data/processed/${dataset}
output_dir=${PWD}/output/${dataset}/${run_name}

mkdir -p $output_dir

python -m simplepatho.run_translation \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --encoder2rnd True \
    --bad_words [CLS] \
    --additional_tokenization_cleanup True \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $data_dir/train.json \
    --validation_file $data_dir/train.json \
    --test_file $data_dir/train.json \
    --source_lang de_COMPLEX \
    --target_lang de_SIMPLE \
    --output_dir $output_dir \
    --logging_dir $output_dir/logs \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate=3e-5 \
    --warmup_ratio 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --max_source_length=512 \
    --max_target_length=512 \
    --num_train_epochs 200 \
    --logging_strategy epoch \
    --evaluation_strategy no \
    --save_strategy no \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --fp16 \
    --report_to wandb \
    --run_name $run_name \
    --group_name $dataset \
    --max_train_samples 10 \
    --max_eval_samples 10 \
    --max_predict_samples 10 \
    "$@"
