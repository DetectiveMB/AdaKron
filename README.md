# AdaKron

## Steps to reproduce our results
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Install the pre-requisites
```console
pip install -e .
```

We also provide the shell scripts for bert-base and roberta-large.

### Quick start
```console
export num_gpus=1
export PYTHONHASHSEED=0
task_name=mnli
model=roberta-large
export output_dir="./models/${model}/${task_name}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name $task_name \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
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
--seed 1234 \
--inference_level 3 \
--weight_decay 0.1 \
--sharing_up 1 \
--sharing_down 0 \
--use_consistency_loss 0

```
* `use_consistency_loss`: Two modes. 
  * `0`: No consistency loss
  * `1`: Use consistency loss


* `sharing_up`: There are two modes (boolean):
  * `True`: Experts defined in both down projection layers
  * `False`: Experts defined only in the weights down projection layer
 

* `sharing_down`: Experts output dimension, we use sharing_down=4

## Notes
The implementation is based on the following projects:  <br>
https://github.com/huggingface/transformers  <br>
https://github.com/microsoft/AdaMix/tree/main <br>
https://github.com/microsoft/LoRA <br>
https://github.com/QingruZhang/AdaLoRA/tree/main
