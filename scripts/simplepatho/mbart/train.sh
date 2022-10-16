# Base configuration
export WANDB_PROJECT=simplepatho
export DATASET=simplepatho
export RUN_NAME=mbart
export MBART_DIR_NAME=output/mbart-simplification-de

# Setup input/output paths
export DATA_DIR=${PWD}/data/processed/${DATASET}
export OUTPUT_DIR=${PWD}/output/${DATASET}/${RUN_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add simplification language tokens to mBART tokenizer and model -- skip if model already exist.
if [ ! -d $MBART_DIR_NAME ]
then
    python scripts/mbart_add_languages.py \
        --model_name_or_path "facebook/mbart-large-cc25" \
        --save_model_to $MBART_DIR_NAME \
        --add_language_tags de_COMPLEX de_SIMPLE \
        --initialize_tags de_DE de_DE
fi

python -m simplepatho.run_translation \
    --model_name_or_path $MBART_DIR_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang de_COMPLEX \
    --target_lang de_SIMPLE \
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
    --num_train_epochs 25 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --fp16 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --group_name $DATASET \
    "$@"
