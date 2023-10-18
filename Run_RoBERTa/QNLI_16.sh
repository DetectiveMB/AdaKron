export num_gpus=1
export PYTHONHASHSEED=0
model=roberta-large

export output_dir="./models/${model}/qnli/ConcatKronecker4Invertiti/128/3e4/16_dim/2_ex/20/seed0/batch64"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
../examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--learning_rate 3e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 1000 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_expert_soup \
--adapter_size 16 \
--num_experts 2 \
--seed 0 \
--inference_level 3 \
--load_best_model_at_end \
--metric_for_best_model "accuracy" \
--sharing_up 0 \
--sharing_down 0 \
--weight_decay 0.1 \
--use_consistency_loss 0

