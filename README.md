# AdaKron and MAdaKron

## Paper links: 

[AdaKron at LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.32/)

[MAdaKron in Knowledge-Based Systems 2025](https://www.sciencedirect.com/science/article/pii/S0950705125021240)

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
| Full Fine-Tuning | 110 | 83.2 | 90.0 | 91.6 | 87.4 | 90.9 | 62.1 | 66.4 | 89.8 | 82.7 |
| Compacter | 1.3 | 81.6 | 89.2 | 87.9 | 85.1 | 88.7 | 58.7 | 59.8 | 86.3 | 79.7 |
| AdaMix Adapter | 0.9 | 84.7 | 91.5 | 92.4 | 87.6 | 92.4 | 62.9 | 74.7 | 89.9 | **84.5** |
| Houlsby Adapter  | 0.9 | 83.1 | 90.6 | 91.9 | 86.8 | 89.9 | 61.5 | 71.8 | 88.6 | 83.0 |
| LoRA  | 0.3 | 82.5 | 89.9 | 91.5 | 86.0 | 90.0 | 60.5 | 71.5 | 85.7 | 82.2 |
| Prefix-tuning | 0.2 | 81.2 | 90.4 | 90.9 | 83.3 | 91.3 | 55.4 | 76.9 | 87.2 | 82.1 |
| BitFit  | 0.1 | 81.4 | 90.2 | 92.1 | 84.0 | 90.4 | 58.8 | 72.3 | 89.2 | 82.3 |
| Vera | 0.04 | 83.1 | 90.5 | 92.3 | 85.9 | 89.9 | 59.0 | 61.0 | 86.8 | 81.1 |
| Pfeiffer Adapter<sub>48</sub> | 0.9 | 83.3 | 91.1 | 92.0 | 87.5 | 90.7 | 60.3 | 67.6 | 89.6 | 82.7 |
| AdaKron<sub>48</sub> | 0.6 | 83.5 | 91.1 | 92.0 | 87.1 | 90.8 | 61.1 | 73.8 | 89.4 | 83.6 |
| MAdaKron<sub>48</sub> | 0.6 | 83.9 | 91.3 | 92.8 | 87.4 | 91.5 | 62.3 | 76.0 | 89.2 | <u>84.3</u> |

<sub>Main results on the GLUE development set with BERT-base. *Avg.* is the average performance across the eight GLUE datasets.</span></sub>


## Cite our works: 

```
@inproceedings{braga2024adakron,
  title={AdaKron: An adapter-based parameter efficient model tuning with kronecker product},
  author={Braga, Marco and Raganato, Alessandro and Pasi, Gabriella},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={350--357},
  year={2024}
}
```



```
@article{braga2025madakron,
  title={MAdaKron: A mixture-of-AdaKron adapters},
  author={Braga, Marco and Raganato, Alessandro and Pasi, Gabriella},
  journal={Knowledge-Based Systems},
  pages={115086},
  year={2025},
  publisher={Elsevier}
}
```

