export OUTPUT_DIR_NAME=output/cochrane/bert2bert
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
    --source_lang en_COMPLEX \
    --target_lang en_SIMPLE \
    --output_dir $OUTPUT_DIR \
    --predict_with_generate \
    --max_source_length=512 \
    --max_target_length=512 \
    --per_device_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --generation_method nucleus \
    --generation_top_p 0.9 \
    --additional_tokenization_cleanup True \
    --overwrite_cache \
    "$@"
