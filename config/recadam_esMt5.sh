python train.py \
    --model_name_or_path LeoCordoba/mt5-small-mlsum \
    --optimizer_name RecAdam \
    --dataset_name de \
    --do_train \
    --do_predict \
    --do_eval \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 8 \
    --output_dir outputs/de/recadam_esMt5 \
    --evaluation_strategy epoch\
    --save_total_limit 3\
    --predict_with_generate True