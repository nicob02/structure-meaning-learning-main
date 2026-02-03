# Reframing linguistic bootstrapping as joint inference using visually-grounded grammar induction models
 
This repository contains all the code and preprocessed data for models and evaluations presented in the paper [*Reframing linguistic bootstrapping as joint inference using visually-grounded grammar induction models*](./writeups/bootstrapping_manuscript_clean.pdf) by Eva Portelance, Siva Reddy, and Timothy J. O'Donnell. 2024.

## Data

All the necessary preprocessed Abstract Scenes data is available in the [preprocessed-data/abstractscenes directory](./preprocessed-data/abstractscenes/). 

The original Abstract Scenes dataset is publicly available at [http://optimus.cc.gatech.edu/clipart/](http://optimus.cc.gatech.edu/clipart/) .

## Model 

The visually-grounded grammar induction model code is available in the folder [./vc-pcfg](./vc-pcfg/). This folder contains a custom modified version of the VC-PCFG model by [Zhao & Titov (2020)](https://aclanthology.org/2020.emnlp-main.354/).

## Evaluation

All the evaluation and result analysis scripts and results from experiment reported in the paper are available in the folder [./analyses/](./analyses/).

## To run code yourself
Use the following to set up a compatible environment. 

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Anaconda package manager (ideally)
- GPU (for faster training, optional but recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/evaportelance/structure-meaning-learning.git
```
If you use anaconda, you can clone our environment using the conda-env.txt file:
```bash
cd structure-meaning-learning
conda create --name myvcpcfgenv --file ./conda-env-graminduct.txt
conda activate myvcpcfgenv
pip install requirements.txt

```

The grammar induction model training requires a custom version of Torch-Struct:
```bash
git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
cd pytorch-struct
pip install -e .
```

### Training the self-supervised image encoder

Note that the pretrained image embeddings are already available in the preprocessed-data folder. If you would like to retrain your own version of the self supervised image encoder, run the following code. This code expects the original Abstract scenes to have the following relative path '../../AbstractScenes_v1.1/RenderedScenes/', but this can by changed in the configs.py file.

```bash
cd structure-meaning-learning/pytorch-simclr
conda activate myvcpcfgenv
python ./simclr.py --dataset 'abstractscenes' --batch-size 100 --num-epochs 500 --cosine-anneal --test-freq 5

```

### Training visually grounded grammar induction models
Use the following commands to train a model.

#### Joint-learning model with self-supervised image embeddings

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int]
```

#### Joint-learning model with visual-labels as embeddings

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int]
```

#### Semantics-first model

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int] --sem_first
```

#### Syntax-first model

```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --log_step 1000 --visual_mode --logger_name [Your logger name] --seed [seed int] --syn_first
```

To train any of these models on the out-of-distribution split where all instances of test verbs are held out, simply add the `--one_shot' flag to call. 

## Traditional Chinese (AbstractScenes) — Experiment 1 only

The repository now supports running Experiment 1 (syntactic structure evaluation) on Traditional Chinese captions. The Chinese captions and image ids live under `preprocessed-data/abstractscenes/` (same location as the English preprocessed data), and the preprocessing below will build a new dataset folder `preprocessed-data/abstractscenes_zh/` in the same format as the English pipeline.

### 1) Prepare the Chinese dataset

This step:
- tokenizes the Traditional Chinese captions
- converts Traditional → Simplified for parsing (HanLP CTB9), then maps spans back to Traditional tokens
- binarizes the constituency trees so each sentence has exactly `L-1` gold spans
- builds `all_caps.json`, `all.id`, `all_caps.text`, and a `vocab_dict.pkl`
- optionally copies the image feature files used by the visual models

Important: The span F1 metric is computed over **Traditional Chinese tokens**. We parse Simplified text to use the HanLP CTB9 model, but we convert tokens back to Traditional before writing `all_caps.json`. Gold spans are stored as token indices for those Traditional tokens, so the evaluation compares predicted spans against gold spans in Traditional-token space.

Install extra dependencies for Chinese parsing if needed:
```bash
pip install hanlp opencc-python-reimplemented nltk
```

```bash
cd structure-meaning-learning/vc-pcfg
python "data preprocessing/as_prepare_zh.py" \
  --input_caps "../preprocessed-data/abstractscenes/all_caps_zh.jsonl" \
  --input_ids "../preprocessed-data/abstractscenes/all.id_zh" \
  --output_dir "../preprocessed-data/abstractscenes_zh" \
  --copy_features_from "../preprocessed-data/abstractscenes" \
  --use_existing_char_spans
```

### 2) Quick smoke test (1000 sentences)

Use the `--limit 1000` flag in preprocessing plus the `--tiny` flag in training to validate the pipeline quickly.

```bash
cd structure-meaning-learning/vc-pcfg
python "data preprocessing/as_prepare_zh.py" \
  --input_caps "../preprocessed-data/abstractscenes/all_caps_zh.jsonl" \
  --input_ids "../preprocessed-data/abstractscenes/all.id_zh" \
  --output_dir "../preprocessed-data/abstractscenes_zh_small" \
  --copy_features_from "../preprocessed-data/abstractscenes" \
  --use_existing_char_spans \
  --limit 1000
```

Then run any model with `--data_path ../preprocessed-data/abstractscenes_zh_small --tiny`.

### 3) Run Experiment 1 only (all four model variants, no Experiment 2)

Use `--skip_syntactic_bootstrapping` to disable Experiment 2’s contrastive verb evaluations. This does **not** remove any of the four Experiment 1 variants (joint-learning, semantics-first, syntax-first, visual-labels). Those are controlled by `--sem_first`, `--syn_first`, and `--encoder_file`, and they still run as usual.

Joint-learning model (self-supervised image embeddings):
```bash
cd structure-meaning-learning/vc-pcfg
python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "all_as-resn-50.npy" \
  --log_step 1000 \
  --visual_mode \
  --logger_name [Your logger name] \
  --seed [seed int] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --skip_syntactic_bootstrapping
```

Semantics-first:
```bash
python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "all_as-resn-50.npy" \
  --log_step 1000 \
  --visual_mode \
  --logger_name [Your logger name] \
  --seed [seed int] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --sem_first \
  --skip_syntactic_bootstrapping
```

Syntax-first:
```bash
python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "all_as-resn-50.npy" \
  --log_step 1000 \
  --visual_mode \
  --logger_name [Your logger name] \
  --seed [seed int] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --syn_first \
  --skip_syntactic_bootstrapping
```

Visual-labels upper bound:
```bash
python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "all_flat_features_gold.npy" \
  --img_dim 756 \
  --log_step 1000 \
  --visual_mode \
  --logger_name [Your logger name] \
  --seed [seed int] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --skip_syntactic_bootstrapping
```

Results for Experiment 1 (span F1) are saved under:
```
[logger_name]/semantic_bootstrapping_results/
```
Each epoch’s CSV includes gold spans and predicted spans (Traditional token indices); aggregate across the 5 seeds to reproduce Table 1 and Figure 4 for Chinese.

### 4) Aggregation script for Table 1 / Figure 4

Use `analyses/analysis_span_f1.py` to reproduce the table and learning curves:
```bash
python analyses/analysis_span_f1.py \
  --run "Joint-learning=/path/to/run1" \
  --run "Semantics-first=/path/to/run2" \
  --run "Syntax-first=/path/to/run3" \
  --run "Visual-labels=/path/to/run4" \
  --last_epoch 29 \
  --max_epoch 30 \
  --switch_epoch 15 \
  --out_table table1.csv \
  --out_plot figure4.png
```

For the 1000‑sentence smoke test, point `--run` to the small runs and set `--max_epoch` to the number of epochs used.

### 5) Nibi SLURM examples

Scripts are provided under `scripts/`:
- `scripts/nibi_preprocess_zh_small.sh`
- `scripts/nibi_train_exp1_zh_small.sh`
- `scripts/nibi_train_exp1_zh_full.sh`

Submit with:
```bash
sbatch scripts/nibi_preprocess_zh_small.sh
sbatch scripts/nibi_train_exp1_zh_small.sh
```

### 4) Optional: Experiment 1 lexical category analysis (Figure 6 / Table 2)

The English pipeline relies on `all_gold_caps.json` for POS tags and a mapping from Penn Treebank tags to broad syntactic categories. For Chinese, the new preprocessing script writes `all_gold_caps.json` using **HanLP CTB9 POS tags**, so you must define an equivalent mapping for CTB9 tags in the analysis scripts.

Steps:
1) Extract predicted preterminal categories for a trained checkpoint:
```bash
cd structure-meaning-learning/vc-pcfg
python ./as_extra_evals.py \
  --model_init [path to checkpoint .pth.tar] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --logger_name [output dir] \
  --out_file pred-parse.json \
  --skip_syntactic_bootstrapping \
  --mode trees
```
2) Build the contingency table:
```bash
python ./as_extra_evals.py \
  --model_init [same checkpoint] \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --logger_name [same output dir] \
  --out_file pred-parse.json \
  --skip_syntactic_bootstrapping \
  --mode cats
```
This creates:
- `[logger_name]/pred-parse.json_df.csv`
- `[logger_name]/pred-parse.json_ct.csv`

3) Update `analyses/analysis-contingency-tables.Rmd` with a CTB9→category mapping (verbs, nouns, adjectives, etc.) and re-run the R analysis to reproduce Figure 6 / Table 2.

Note: CTB9 tags differ from English Penn tags (e.g., `VV`, `VA`, `VC`, `VE` for verbs; `NN`, `NR`, `NT` for nouns). The mapping is language-specific and must be updated for Chinese.



## Citation

Please cite the following paper:
```
@article{portelance2024reframing,
  title={Reframing linguistic bootstrapping as joint inference using
visually-grounded grammar induction models},
  author={Portelance, Eva and Reddy, Siva and O'Donnell, Timothy J.},
  year={2024},
  journal={ArXiv preprint: 2406.11977},
  url={https://arxiv.org/abs/2406.11977}
}
```

Please also cite the VC-PCFG paper and the Abstract Dataset papers:

```
@inproceedings{zhao-titov-2020-visually,
    title = "Visually Grounded Compound {PCFG}s",
    author = "Zhao, Yanpeng  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.354",
    pages = "4369--4379"
}
```
```
@INPROCEEDINGS{zitnick2013learning,
  author={Zitnick, C. Lawrence and Parikh, Devi and Vanderwende, Lucy},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision}, 
  title={Learning the Visual Interpretation of Sentences}, 
  year={2013},
  pages={1681-1688},
  url={https://ieeexplore.ieee.org/document/6751319}}

@INPROCEEDINGS{zitnick2013bringing,
  author={Zitnick, C. Lawrence and Parikh, Devi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}, 
  title={Bringing Semantics into Focus Using Visual Abstraction}, 
  year={2013},
  pages={3009-3016},
  url={https://ieeexplore.ieee.org/document/6619231}}
```

If you use the pytorch-simclr code to retrain the image encoder, please cite the repo it is forked from :

```
@article{
  silva2020exploringsimclr,
  title={Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations},
  author={Silva, Thalles Santos},
  journal={https://sthalles.github.io},
  year={2020}
  url={https://sthalles.github.io/simple-self-supervised-learning/}
}
```
