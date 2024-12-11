# Adapating DeBERTaV3 with AdaKron for Question Answering (QA) task

The folder contains the implementation of AdaKron in DeBERTaV3. 

## Setup Environment

### Install the pre-requisites
Install dependencies: 
```bash
pip install -r requirements.txt
```

### Running AdaKron with DeBERTa-v3-base on SQuAD (QA task)

```bash
python -m torch.distributed.run --nproc_per_node=1 \
examples/question-answering/run_qa.py \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad \
--apply_adapter \
--adapter_size 32 \
--adapter_type 'pfeiffer' \
--init_warmup 5000 --final_warmup 25000 --mask_interval 100 \
--do_train --do_eval  \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 10 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy epoch \
--save_strategy epoch \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 0 \
--output_dir ./output/debertav3-base/squadv1/32/seed0 \
--overwrite_output_dir \
--disable_tqdm True \
--save_total_limit 1 \
```
