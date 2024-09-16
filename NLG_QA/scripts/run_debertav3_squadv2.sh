# ################################################ 64 

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 64 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 0 \
# --output_dir ./output/debertav3-base/squadv2/64/seed0 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 64 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --output_dir ./output/debertav3-base/squadv2/64/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 64 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --output_dir ./output/debertav3-base/squadv2/64/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 64 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --output_dir ./output/debertav3-base/squadv2/64/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \




# ################################################ 32

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 0 \
# --output_dir ./output/debertav3-base/squadv2/32/seed0 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --output_dir ./output/debertav3-base/squadv2/32/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --output_dir ./output/debertav3-base/squadv2/32/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --output_dir ./output/debertav3-base/squadv2/32/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# ################################################ 16

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 0 \
# --output_dir ./output/debertav3-base/squadv2/16/seed0 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --output_dir ./output/debertav3-base/squadv2/16/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --output_dir ./output/debertav3-base/squadv2/16/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --output_dir ./output/debertav3-base/squadv2/16/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \




# ################################################ 8

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 0 \
# --output_dir ./output/debertav3-base/squadv2/8/seed0 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --output_dir ./output/debertav3-base/squadv2/8/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad_v2 \
# --apply_adapter True \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval --version_2_with_negative \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 12 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --output_dir ./output/debertav3-base/squadv2/8/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
examples/question-answering/run_qa.py \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_adapter True \
--adapter_size 8 \
--adapter_type 'pfeiffer' \
--init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 12 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy epoch \
--save_strategy epoch \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 4321 \
--output_dir ./output/debertav3-base/squadv2/8/seed4321 \
--overwrite_output_dir \
--disable_tqdm True \
--save_total_limit 1 \






################################################################
#--root_output_dir ./output/debertav3-base/squadv2/64/seed0 \
# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --root_output_dir ./output/debertav3-base/squadv1/32/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \


#--output_dir ./output_nlg \
#--version_2_with_negative

################################à
#python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --root_output_dir ./output/debertav3-base/squadv1/8/4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 42 \
# --root_output_dir ./output/debertav3-base/squadv1/8/seed42 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 0 \
# --root_output_dir ./output/debertav3-base/squadv1/8/seed0 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# ###########################################
# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 8 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --root_output_dir ./output/debertav3-base/squadv1/8/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --root_output_dir ./output/debertav3-base/squadv1/32/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 1234 \
# --root_output_dir ./output/debertav3-base/squadv1/16/seed1234 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# ###########################################

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 64 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --root_output_dir ./output/debertav3-base/squadv1/64/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 32 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --root_output_dir ./output/debertav3-base/squadv1/32/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \

# python -m torch.distributed.run --master_port=29600 --nproc_per_node=1 \
# examples/question-answering/run_qa.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --dataset_name squad \
# --apply_adapter \
# --adapter_size 16 \
# --adapter_type 'pfeiffer' \
# --init_warmup 5000 --final_warmup 50000 --mask_interval 100 \
# --do_train --do_eval  \
# --max_seq_length 384 --doc_stride 128 \
# --per_device_train_batch_size 16 \
# --learning_rate 1e-3 \
# --num_train_epochs 3 \
# --warmup_steps 1000 --per_device_eval_batch_size 128 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --logging_steps 300 \
# --tb_writter_loginterval 300 \
# --report_to tensorboard \
# --seed 4321 \
# --root_output_dir ./output/debertav3-base/squadv1/16/seed4321 \
# --overwrite_output_dir \
# --disable_tqdm True \
# --save_total_limit 1 \