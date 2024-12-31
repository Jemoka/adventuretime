#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a6000:1
#SBATCH --job-name=houjun-adventure-e0_implement_me
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e0_implement_me.log
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

cd .
source .venv/bin/activate

python main.py implement_me

