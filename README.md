# Patient-friendly Clinical Notes: Towards a new Text Simplification Dataset

TSAR2022

## Computational Environment

```sh
conda update -f environment.yml
conda activate simple-patho

pip install -r requirements-dev.txt
pip install -e .

# if on GPU, install CUDA toolkit
conda install cudatoolkit=11.6.0
```


## Data Format

Data is expected to be split into train/val/test sets and to have one sample per line in source/target.

```sh
tree data/raw/$YOUR_DATASET
├── test.source
├── test.target
├── train.source
├── train.target
├── val.source
└── val.target
```

For training seq2seq models, convert the data into jsonlines format.

```sh
python -m scripts.convert_data \
    --raw_path data/raw/$YOUR_DATASET \
    --out_path data/processed/$YOUR_DATASET

# Should give
tree data/processed/$YOUR_DATASET
├── test.json
├── train.json
└── val.json
```

## Corpus Statistics

```sh
python -m simplepatho.corpus_statistics \
    --data_path data/raw/$YOUR_DATASET \
    --language [english|german] \
    --spacy_model [en_core_web_sm|de_core_news_sm]
```

## Model Training and Evaluation

In the paper, we run experiments with three models: Bert2Bert, Bert2Share, and mBART. Please refer to the files in [scripts/simplepatho](./scripts/simplepatho) for training/inference parameters and how to use your own dataset.

```sh
# Training
sh scripts/simplepatho/{bert2bert,bert2share,mbart}/train.sh
# Inference
sh scripts/simplepatho/{bert2bert,bert2share,mbart}/predict.sh
```

Model predictions can be evaluated with below command.

```sh
# --sys_paths takes multiple files, each being the predictions of one system.
# Below, the first run corresponds to the identity baseline.
python -m simplepatho.evaluate \
    --orig_path data/processed/$YOUR_DATASET/test.source \
    --refs_path data/processed/$YOUR_DATASET/test.target \
    --sys_paths \
        data/processed/$YOUR_DATASET/test.source \
        output/$YOUR_DATASET/bert2bert/predictions_test.txt \
        output/$YOUR_DATASET/bert2bert-shared/predictions_test.txt \
        output/$YOUR_DATASET/mbart/predictions_test.txt \
    --compact \
    --calculate_metrics R-1 R-2 R-L BLEU SARI "|y|_w" "CMP_w" "CMP_s" "Nov. (n=1)" "Nov. (n=2)" \
    --n_jobs 4
```

## Experiments on Cochrane Data (English, Medical)

The code in this repository can be used to reproduce the BART-based model described in [Paragraph-level Simplification of Medical Texts](https://aclanthology.org/2021.naacl-main.395) (Devaraj et al., NAACL 2021). We reimplement the [official code](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts) with following changes:

- Evaluation is based on [easse](https://github.com/feralvam/easse/) (BLEU, SARI) and [rouge_score](https://pypi.org/project/rouge-score/) (ROUGE). The original evaluation script is [not available](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts/issues/4) as of 10/2022.
- We use the Huggingface seq2seq trainer rather than PyTorch lightning. See [`simplepatho/run_translation.py`](./simplepatho/run_translation.py).
- No support for unlikelihood loss (corresponds to `bart-no-ul` in Devaraj et al., 2021)

Download and preprocess raw data:

```sh
make cochrane
```

Train models:

```sh
sh scripts/cochrane/{bart,bart-xsum,bert2bert,mbart}/train.sh
sh scripts/cochrane/{bart,bart-xsum,bert2bert,mbart}/predict.sh
```

Evaluate predictions:

```sh
python -m simplepatho.evaluate \
    --orig_path data/raw/cochrane/test.source \
    --refs_path data/raw/cochrane/test.target \
    --sys_paths \
        data/raw/cochrane/test.source \
        data/raw/cochrane/test.target \
        output/cochrane/bart/predictions_test.txt \
        output/cochrane/bart-xsum/predictions_test.txt \
        output/cochrane/bert2bert/predictions_test.txt \
        output/cochrane/mbart/predictions_test.txt \
    --compact \
    --calculate_metrics R-1 R-2 R-L BLEU SARI "|y|_w" "CMP_w" "Nov. (n=1)" "Nov. (n=2)" \
    --n_jobs 6
```

Results are given in the table below.

| | R-1 | R-2 | R-L | BLEU | SARI | \|y\|\_w | CMP\_w | Nov. (n=1) | Nov. (n=2) |
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| `gold` | - | - | - | - | - | 203.52 | 0.47 | 0.37 | 0.69 |
| `identity` | 43.82 | 19.63 | 41.34 | 14.47 | 9.62 | 411.42 | 0.00 | 0.00 | 0.00 |
| `bart-no-ul`* | 40.00 | 15.00 | 37.00 | 44.00 | 38.00 | 228.27 | - | 0.05 | 0.11 |
|_Reproduction results._|
| `bart` | 45.36 | 18.43 | 42.42 | 16.22 | 35.24 | 227.93 | 0.42 | 0.07 | 0.15 |
| `bart-xsum` (corresponds to `bart-no-ul`) | 45.13 | 18.38 | 42.27 | 16.48 | 34.23 | 240.92 | 0.39 | 0.05 | 0.14 |
| `bert2bert` | 31.84 | 5.71 | 29.54 | 3.41 | 38.40 | 213.88 | 0.44 | 0.60 | 0.92 |
| `mbart` | 43.77 | 19.16 | 41.14 | 14.59 | 20.64 | 372.61 | 0.08 | 0.04 | 0.06 |

_\*Taken from [(Devaraj et al. 2021)](https://doi.org/10.18653/v1/2021.naacl-main.395)_, Table 6.
