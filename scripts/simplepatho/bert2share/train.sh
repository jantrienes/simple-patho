# Base configuration
export WANDB_PROJECT=simplepatho
export DATASET=simplepatho
export RUN_NAME=bert2share

# Setup input/output paths
export DATA_DIR=${PWD}/data/processed/${DATASET}
export OUTPUT_DIR=${PWD}/output/${DATASET}/${RUN_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python -m simplepatho.run_translation \
    --bert2bert True \
    --model_name_or_path bert-base-multilingual-cased \
    --tie_encoder_decoder True \
    --bad_words [CLS] \
    --additional_tokenization_cleanup True \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --source_lang de_COMPLEX \
    --target_lang de_SIMPLE \
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
    --num_train_epochs 25 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --fp16 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --group_name $DATASET \
    "$@"
