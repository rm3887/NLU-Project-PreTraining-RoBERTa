#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=96000
#SBATCH --time=1:00:00
#SBATCH --job-name="nlu_final_project"
#SBATCH --mail-user=rm3887@nyu.edu

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=rm3887
source activate /scratch/${NETID}/nlu/env

GLUE_DIR=/scratch/${NETID}/nlu/data

TASK_NAME=Reddit
cd /scratch/${NETID}/nlu/code/transformers
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH
python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --data_dir ${GLUE_DIR}/${TASK_NAME} \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --output_dir /scratch/${NETID}/nlu/${TASK_NAME}_run/ \
    --overwrite_output_dir
