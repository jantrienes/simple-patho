# Base configuration
export DATASET=cochrane
export RUN_NAME=bert2bert

# Setup input/output paths
export DATA_DIR=${PWD}/data/processed/${DATASET}
export OUTPUT_DIR=${PWD}/output/${DATASET}/${RUN_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --bad_words [CLS] \
    --additional_tokenization_cleanup True \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang en_COMPLEX \
    --target_lang en_SIMPLE \
    --output_dir $OUTPUT_DIR \
    --max_source_length=512 \
    --max_target_length=512 \
    --per_device_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --additional_tokenization_cleanup True \
    --overwrite_cache \
    "$@"
