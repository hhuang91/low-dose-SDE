#!/bin/bash
#SBATCH --array 1-1
#SBATCH --cpus-per-task=16
#SBATCH --gres gpu:3
#SBATCH --job-name LDSDE
#SBATCH --partition gpu
#SBATCH --time 120:0:0
#SBATCH -D /home/hhuang91/Projects/DeNoise
#SBATCH --output /home/hhuang91/Projects/DeNoise/logs/training.log
#SBATCH --mail-type end
#SBATCH --mail-user hhuang91@jhu.edu
source /home/hhuang91/.bashrc
export PATH="$PATH:/home/hhuang91/miniconda3/bin"

source activate DeNoise

python --version
python -m torch.distributed.launch --nproc_per_node=3 "TemplateDDP_cluster.py" --ngpu=3