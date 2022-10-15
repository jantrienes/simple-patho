export OUTPUT_DIR_NAME=output/cochrane/bert2bert
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/processed/cochrane
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --bert2bert True \
    --model_name_or_path bert-base-cased \
    --tie_encoder_decoder True \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang en_COMPLEX \
    --target_lang en_SIMPLE \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --learning_rate=3e-5 \
    --warmup_ratio 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --max_source_length=512 \
    --max_target_length=512 \
    --num_train_epochs 50 \
    --logging_strategy steps \
    --logging_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 120 \
    --save_strategy steps \
    --save_steps 120 \
    --load_best_model_at_end True \
    --save_total_limit 5 \
    --fp16 \
    "$@"
