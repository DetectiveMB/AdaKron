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

## Detailed Results on GLUE 

| **Model** | **# Params (M)** | **MNLI** | **QNLI** | **SST2** | **QQP** | **MRPC** | **CoLa** | **RTE** | **STS-B** | **Avg.** |
|-----------|-----------------|----------|----------|----------|---------|----------|----------|----------|-----------|----------|
| Full Fine-Tuning [Wang2022AdaMixMF] | 110 | 83.2 | 90.0 | 91.6 | 87.4 | 90.9 | 62.1 | 66.4 | 89.8 | 82.7 |
| UNIPELT [Wang2022AdaMixMF] | 1.4 | 83.9 | 90.5 | 91.5 | 85.5 | 90.2 | 58.6 | 73.7 | 88.9 | 83.5 |
| Compacter (repr.) | 1.3 | 81.6 | 89.2 | 87.9 | 85.1 | 88.7 | 58.7 | 59.8 | 86.3 | 79.7 |
| <span style="color:blue">Compacter++ (repr.)</span> | <span style="color:blue">1.3</span> | <span style="color:blue">80.0</span> | <span style="color:blue">89.8</span> | <span style="color:blue">91.6</span> | <span style="color:blue">83.7</span> | <span style="color:blue">89.5</span> | <span style="color:blue">57.8</span> | <span style="color:blue">60.5</span> | <span style="color:blue">86.2</span> | <span style="color:blue">79.9</span> |
| <span style="color:blue">(IA)<sup>3</sup> (repr.)</span> | <span style="color:blue">1.2</span> | <span style="color:blue">79.4</span> | <span style="color:blue">88.8</span> | <span style="color:blue">92.3</span> | <span style="color:blue">82.7</span> | <span style="color:blue">90.1</span> | <span style="color:blue">59.9</span> | <span style="color:blue">65.4</span> | <span style="color:blue">86.8</span> | <span style="color:blue">80.7</span> |
| <span style="color:blue">MoLE-Adapter [li2023mixture]</span> | <span style="color:blue">1.2</span> | <span style="color:blue">84.3</span> | <span style="color:blue">93.0</span> | <span style="color:blue">92.7</span> | <span style="color:blue">87.8</span> | <span style="color:blue">90.4</span> | <span style="color:blue">61.5</span> | <span style="color:blue">70.4</span> | <span style="color:blue">88.7</span> | <span style="color:blue">83.6</span> |
| <span style="color:blue">UNIPELT (AP) [Mao2021UniPELTAU]</span> | <span style="color:blue">1.1</span> | <span style="color:blue">83.4</span> | <span style="color:blue">90.8</span> | <span style="color:blue">91.9</span> | <span style="color:blue">86.7</span> | <span style="color:blue">90.3</span> | <span style="color:blue">61.2</span> | <span style="color:blue">71.8</span> | <span style="color:blue">88.9</span> | <span style="color:blue">83.1</span> |
| AdaMix Adapter [Wang2022AdaMixMF] | 0.9 | 84.7 | 91.5 | 92.4 | 87.6 | 92.4 | 62.9 | 74.7 | 89.9 | **84.5** |
| Houlsby Adapter [Wang2022AdaMixMF] | 0.9 | 83.1 | 90.6 | 91.9 | 86.8 | 89.9 | 61.5 | 71.8 | 88.6 | 83.0 |
| LoRA [Wang2022AdaMixMF] | 0.3 | 82.5 | 89.9 | 91.5 | 86.0 | 90.0 | 60.5 | 71.5 | 85.7 | 82.2 |
| Prefix-tuning [Wang2022AdaMixMF] | 0.2 | 81.2 | 90.4 | 90.9 | 83.3 | 91.3 | 55.4 | 76.9 | 87.2 | 82.1 |
| BitFit [Wang2022AdaMixMF] | 0.1 | 81.4 | 90.2 | 92.1 | 84.0 | 90.4 | 58.8 | 72.3 | 89.2 | 82.3 |
| <span style="color:blue">Vera (repr.)</span> | <span style="color:blue">0.04</span> | <span style="color:blue">83.1</span> | <span style="color:blue">90.5</span> | <span style="color:blue">92.3</span> | <span style="color:blue">85.9</span> | <span style="color:blue">89.9</span> | <span style="color:blue">59.0</span> | <span style="color:blue">61.0</span> | <span style="color:blue">86.8</span> | <span style="color:blue">81.1</span> |
| <span style="color:blue">Hadamard-Adapter [Hadamard_adapter]</span> | <span style="color:blue">0.03</span> | <span style="color:blue">80.4</span> | <span style="color:blue">89.7</span> | <span style="color:blue">92.4</span> | <span style="color:blue">85.9</span> | <span style="color:blue">90.2</span> | <span style="color:blue">58.4</span> | <span style="color:blue">71.9</span> | <span style="color:blue">88.5</span> | <span style="color:blue">82.2</span> |
| Pfeiffer Adapter<sub>48</sub> | 0.9 | 83.3 | 91.1 | 92.0 | 87.5 | 90.7 | 60.3 | 67.6 | 89.6 | 82.7 |
| AdaKron<sub>48</sub> | 0.6 | 83.5 | 91.1 | 92.0 | 87.1 | 90.8 | 61.1 | 73.8 | 89.4 | 83.6 |
| Full MAdaKron<sub>48</sub> | 0.6 | 83.9 | 91.3 | 92.8 | 87.4 | 91.5 | 62.3 | 76.0 | 89.2 | <u>84.3</u> |
| Partial MAdaKron<sub>48</sub> | 0.6 | 83.9 | 91.1 | 92.3 | 87.6 | 91.1 | 61.8 | 74.2 | 89.4 | 83.9 |

<sub><span style="color:blue">Main results on the GLUE development set with BERT-base. *Avg.* is the average performance across the eight GLUE datasets. **Bold** marks the best average, and <u>underline</u> marks the second best.</span></sub>

