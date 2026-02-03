#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

cd $HOME/structure-meaning-learning-main/vc-pcfg

python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "all_as-resn-50.npy" \
  --visual_mode \
  --logger_name $HOME/structure-meaning-learning-main/runs/zh_full_joint_s1018 \
  --seed 1018 \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --skip_syntactic_bootstrapping
