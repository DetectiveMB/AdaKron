#!/bin/bash
#SBATCH -J runNLG_webnlg_solo_adakron_seed0
#SBATCH -o runNLG_webnlg_solo_adakron_seed0.%J.txt
#SBATCH -e runNLG_webnlg_solo_adakron_seed0.%J.txt
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4         # 32 tasks per node
#SBATCH --time=23:50:00               # time limits: 1 hour
#SBATCH --account=IscrC_EPeRLLM       # account name
#SBATCH --gres=gpu:1 
#SBATCH --partition=boost_usr_prod #lrd_all_serial #boost_usr_prod # partition name
#SBATCH --qos=normal #boost_qos_dbg
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL


####################
 ##########SBATCH --gres=gpu:1        # 1 gpus per node out of 4

#/leonardo/home/userexternal/araganat/cache/
 #/leonardo_work/IscrC_EPeRLLM/ #/leonardo/home/userexternal/araganat/cache/ #/leonardo_work/IscrC_EPeRLLM/local/
 #home/
#export HOME=$LOCAL_HOME

#module load profile/deeplrn cineca-ai
module load singularity/3.8.7
#python3 -m venv venv
#source venv4/bin/activate

LOCAL_SCRATCH=/leonardo_work/IscrC_EPeRLLM/local/
LOCAL_HOME=/leonardo_work/IscrC_EPeRLLM/home/

export HOME=$LOCAL_HOME

export TRANSFORMERS_CACHE=$LOCAL_SCRATCH 
export TORCH_CACHE=$LOCAL_SCRATCH 
export SENTENCE_TRANSFORMERS_HOME=$LOCAL_SCRATCH 
export MPLCONFIGDIR=$LOCAL_SCRATCH

export HF_HOME=$LOCAL_SCRATCH
export HUGGINGFACE_HUB_CACHE=$LOCAL_SCRATCH
export HF_DATASETS_CACHE=$LOCAL_SCRATCH
export HF_MODULES_CACHE=$LOCAL_SCRATCH
export hf_cache_home=$LOCAL_SCRATCH
export XDG_CACHE_HOME=$LOCAL_SCRATCH

export PYTHONHASHSEED=0
export output_dir="/leonardo_work/IscrC_EPeRLLM/NLG/"

#python -u /leonardo_work/IsqcrC_PersLLMs/down_data.py

#singularity exec --nv --writable-tmpfs -B /leonardo_work/IscrC_EPeRLLM/,/leonardo_work/IscrC_EPeRLLM/local/ /leonardo_work/IscrC_EPeRLLM/sandpytorch2212Braga python -u /leonardo_work/IscrC_EPeRLLM/run_glueBraga.py \
#. ./venv/bin/activate

export seed=0

singularity exec --nv --writable-tmpfs -B /leonardo_work/IscrC_EPeRLLM/,/leonardo_work/IscrC_EPeRLLM/local/ /leonardo_work/IscrC_EPeRLLM/sandpytorch2212Braga python -m torch.distributed.launch --nproc_per_node=1 /leonardo_work/IscrC_EPeRLLM/NLG/src/gpt2_ft.py \
--train_data /leonardo_work/IscrC_EPeRLLM/NLG/data/webnlg_challenge_2017/train.jsonl \
--valid_data /leonardo_work/IscrC_EPeRLLM/NLG/data/webnlg_challenge_2017/train.jsonl \
--train_batch_size 8 \
--grad_acc 1 \
--valid_batch_size 4 \
--seq_len 512 \
--model_card gpt2.md \
--init_checkpoint /leonardo_work/IscrC_EPeRLLM/NLG/pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
--platform local \
--clip 0.0 \
--lr 0.0002 \
--weight_decay 0.01 \
--correct_bias \
--adam_beta2 0.999 \
--scheduler linear \
--warmup_step 2500 \
--max_epoch 25 \
--eval_interval 5000 \
--save_interval 5000 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--label_smooth 0.1 \
--work_dir /leonardo_work/IscrC_EPeRLLM/NLG/trained_models/GPT2_M/webnlg/$seed/solo_makron \
--random_seed $seed \
--n_experts 1 \
--adamix_only 1 \
--lora_only 0 \
--share_A 0 \
--share_B 1

#bash run_eval_e2e.sh --seed $seed --n_experts $n_experts