#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load python/3.10 cuda/12.2

source ~/venvs/graminduct/bin/activate

if [ ! -d "$HOME/pytorch-struct" ]; then
  git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git ~/pytorch-struct
  cd ~/pytorch-struct
  pip install -e .
fi

cd /projects/def-eporte2/nicob0/structure-meaning-learning-main/vc-pcfg

python ./as_train.py \
  --num_epochs 5 \
  --encoder_file "all_as-resn-50.npy" \
  --visual_mode \
  --logger_name /projects/def-eporte2/nicob0/structure-meaning-learning-main/runs/zh_small_joint_s1018 \
  --seed 1018 \
  --data_path "../preprocessed-data/abstractscenes_zh_small" \
  --skip_syntactic_bootstrapping \
  --tiny
