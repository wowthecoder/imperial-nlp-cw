# 🧠 Patronizing and Condesceding Language Binary Classification
This repository contains the code, notebooks and data used for the Imperial College London COMP60035 3rd‑year Natural Language Processing coursework. The coursework is [SemEval-2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022/home) subtask 1. The project explores how modern transformer‑based models can be adapted to detect patronising or condescending language (PCL). It includes a range of experiments—from simple baselines to state‑of‑the‑art models—and evaluates the impact of data augmentation, oversampling, weighted loss functions and ablations. 

## 🗂️ Repository overview
The project is organised into top‑level directories that group related experiments. A high‑level diagram of the directory structure is shown below. 

### Top‑level directories
- `BestModel/` – contains the final notebook used to train the best‑performing model and the development/test predictions for submission.
- `ablation_studies/` – notebooks for exploring how individual components (data augmentation, oversampling, weighted loss, etc.) affect performance.
- `data_augmentation/` – code and data used to increase the size of the training set through augmentation techniques.
- `deberta_experiments/` – experiments using the DeBERTa transformer architecture (e.g. deberta‑pcl.ipynb, deberta‑prompt‑pcl.ipynb, deberta‑v3‑pcl.ipynb).
- `eda_diagrams/` – exploratory data analysis notebooks and visualisations.
- `ensemble/` – experiments combining predictions from multiple models.
- `hatebert_experiments/` and `modernbert_experiments/` – experiments using other BERT‑based models.
- `roberta_experiments/` – experiments using RoBERTa, including roberta‑prompt‑pcl.ipynb.
- `train/` and `test/` – directories containing training and testing data.
- `root` – miscellaneous scripts and data such as `dont_patronize_me.py` (CSV data loader), `dontpatronizeme_pcl.tsv` (full original dataset), `eda_pcl.ipynb` (Exploratory Data Analysis)

## Key folders and notebooks
### 🏆 BestModel
The `BestModel` directory contains a single notebook, `best_model.ipynb`, together with `dev.txt` and `test.txt`. This notebook does the following in order:
1. Load the full dataset and split it into train and dev(validation) sets according to the provided official splits
2. Preprocesses the text samples to expand contractions and strip HTML tags and whitespace. Add keyword and country code into the text of each sample.
3. Append the augmented training data onto the train set (for details, see the section below on data augmentation)
4. Uses the HuggingFace Trainer API and transformers library to train a RoBERTA-base model on the augmented training data. PCL class F1 is used as the validation metric during training and for early stopping. Evals performed every 500 steps starting from the 1000th step, early stopping patience is 5 eval events (2500 steps)
5. After training, evaluate on dev set with the best model and generate classification report and confusion matrix
6. Generates predictions for dev set into the file `dev.txt` for coursework submission
7. Append the augmented dev data onto the dev set, the combine all sets (train + augmented train + dev + augmented dev) into a final training set
8. Train a fresh RoBERTa model on this final training set and generate test set predictions into `test.txt` for coursework submission

### ➕ Data augmentation
`data‑augmentation‑pcl.ipynb` – a notebook that applies 4 data augmentation strategies to the PCL dataset: 
1. Duplication (2x per sample)
2. Synonym replacement with WordNet (1x per sample)
3. Back translation, English to German to English using Helsinki-NLP Opus MT models (2x per sample)
4. Paraphrasing via Gemini API using Gemini-2.5-Flash (2x per sample)

This notebook generates new training examples for the minority class to combat class imbalance and saves the augmented sets to CSV files. It generates 7 new samples for each original PCL positive sample in the dataset. It is applied separately to the train and dev splits.

`add_duplicate.py` – I wanted to produce 8x new samples instead of 7x so this is a quick script to duplicate all the samples 1x and add to the output CSV from the notebook above.

`augmented_train_data.csv` and `augmented_dev_data.csv` are data files created by the augmentation notebook.

### 🔬 Ablation studies

The `ablation_studies` folder contains several notebooks designed to isolate and understand the contribution of individual techniques. All trainings are run for 2 epochs. The directory listing shows the range of notebooks available:

1. `baseline.ipynb` – Trains a RoBERTa-base model on the official train split without any add-ons at all. Best PCL class F1: **0.5496** 
2. ✅ `preprocess.ipynb – baseline + preprocessing + addition of keywords and country code. Best PCL class F1: **0.5788**
3. 🛑`parameters.ipynb` – No. 2 + some hyperparameters (e.g. weight decay and cosine LR scheduler) + early stopping. Best PCL class F1: **0.5693** 
4. ✅ `oversampling.ipynb` – Oversample PCL positive class by 9x (each sample duplicated 8x). Best PCL class F1: **0.6050**
5. ✅ `augmented_oversampling.ipynb` – Synonym replacement(1x), Back Translation (2x) and Duplication (5x). Best PCL class F1: **0.6078**
6. ✅ `gemini_oversampling.ipynb` – Gemini Paraphrasing (2x) and Duplication (5x). Best PCL class F1: **0.6083**
7. 🛑 `weighted_loss.ipynb` – No. 2 + weighted cross entropy loss. Best PCL class F1: **0.5831**

✅ for implemented in the best model notebook. \
🛑 for not implemented in the best model notebook.

## 📈 Results on original dev set
| Class        | Precision | Recall | F1-score | Support |
|--------------|----------|--------|----------|---------|
| No-PCL       | 0.9622   | 0.9546 | 0.9584   | 1895    |
| **PCL**          | 0.5981   | 0.6432 | **0.6199**   | 199     |
| Accuracy |          |        | 0.9250 | 2094 |
| Macro Avg| 0.7802   | 0.7989 | 0.7891   | 2094    |
| Weighted Avg | 0.9276 | 0.9250 | 0.9262 | 2094 |

## ▶️ Getting started

To run the notebooks in this repository you will need Python 3.8+ and the dependencies listed in requirements.txt. Clone the repository, create a virtual environment, install the dependencies and open the notebooks in Jupyter Lab or VS Code. You may also need to download the dontpatronizeme_pcl.tsv dataset file (included in the repo) and adjust file paths in the notebooks accordingly.

Clone the repository and install dependencies:
```bash
git clone https://github.com/wowthecoder/imperial-nlp-cw.git
cd imperial-nlp-cw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch a Jupyter session:
```
jupyter lab
```

## 📝 Note
Some notebooks are written to be execute on Kaggle. For example, the `best_model.ipynb` notebook is run on Kaggle to utilise the powerful GPU. 
