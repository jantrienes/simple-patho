export MBART_DIR_NAME=output/mbart-simplification-en
export OUTPUT_DIR_NAME=output/cochrane/mbart-lang-toks
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/processed/cochrane
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add simplification language tokens to mBART tokenizer and model -- skip if model already exist.
if [ ! -d $MBART_DIR_NAME ]
then
    python scripts/mbart_add_languages.py \
        --model_name_or_path "facebook/mbart-large-cc25" \
        --save_model_to $MBART_DIR_NAME \
        --add_language_tags en_COMPLEX en_SIMPLE \
        --initialize_tags en_XX en_XX
fi

python -m simplepatho.run_translation \
    --model_name_or_path $MBART_DIR_NAME \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang en_COMPLEX \
    --target_lang en_SIMPLE \
    --use_custom_mbart_tokenizer True \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --overwrite_output_dir \
    --overwrite_cache True \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --learning_rate=3e-5 \
    --warmup_ratio 0.1 \
    --max_source_length=1024 \
    --max_target_length=1024 \
    --num_train_epochs 100 \
    --logging_strategy steps \
    --logging_steps 25 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --load_best_model_at_end True \
    --save_total_limit 5 \
    --fp16 \
    "$@"
