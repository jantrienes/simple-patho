export OUTPUT_DIR_NAME=output/cochrane/bart
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/processed/cochrane
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang en_XX \
    --target_lang en_XX \
    --output_dir $OUTPUT_DIR \
    --predict_with_generate \
    --max_source_length=1024 \
    --max_target_length=1024 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --overwrite_cache \
    "$@"
