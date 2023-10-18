#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 05:00:00
#SBATCH --partition=GPUampere,GPUhopper

eval "$(conda shell.bash hook)"
conda activate simple-patho

export WANDB_PROJECT=simplepatho-clamog
dataset=d2h-v1-aligned-para

if [ $# -lt 2 ]; then
  echo "Usage: $0 arg1 arg2 [other args...]"
  exit 1
fi
model_name_or_path="$1"
run_name="$2-debug"
shift 2

data_dir=${PWD}/data/processed/${dataset}
output_dir=${PWD}/output/${dataset}/${run_name}

mkdir -p $output_dir

# Setting the max sequence length to 504 (the next smallest multiple of 8 below 512)
# This is because of a weird behavior of the roberta embedding layer when the padding token is not equal to 1.
# See: https://github.com/huggingface/transformers/issues/15292
python -m simplepatho.run_translation \
    --model_name_or_path $model_name_or_path \
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
    --max_source_length=510 \
    --max_target_length=510 \
    --num_train_epochs 200 \
    --logging_strategy epoch \
    --evaluation_strategy no \
    --save_strategy no \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --fp16 \
    --report_to none \
    --run_name $run_name \
    --group_name $dataset \
    --max_train_samples 10 \
    --max_eval_samples 10 \
    --max_predict_samples 10 \
    --dropout 0 \
    "$@"
