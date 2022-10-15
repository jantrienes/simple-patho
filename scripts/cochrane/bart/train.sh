export OUTPUT_DIR_NAME=output/cochrane/bart
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/processed/cochrane
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang en_XX \
    --target_lang en_XX \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --learning_rate=3e-5 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --max_source_length=1024 \
    --max_target_length=1024 \
    --num_train_epochs 1 \
    --logging_strategy steps \
    --logging_steps 25 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --fp16 \
    "$@"
