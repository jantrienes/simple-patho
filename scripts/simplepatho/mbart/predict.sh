# Base configuration
export DATASET=simplepatho
export RUN_NAME=mbart

# Setup input/output paths
export DATA_DIR=${PWD}/data/processed/${DATASET}
export OUTPUT_DIR=${PWD}/output/${DATASET}/${RUN_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang de_COMPLEX \
    --target_lang de_SIMPLE \
    --use_custom_mbart_tokenizer True \
    --output_dir $OUTPUT_DIR \
    --max_source_length=1024 \
    --max_target_length=1024 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --overwrite_cache \
    --report_to none \
    "$@"
