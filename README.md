# Explainable Chest X-ray Pneumonia Classification

An end-to-end deep learning research project using PyTorch, DenseNet-121, Grad-CAM, SHAP, and perturbation-based shortcut audits.

> **Research demonstration only. Not a medical device or diagnostic tool.**

## Project Overview

This project explores whether a deep learning classifier can distinguish pneumonia-labeled chest X-rays from normal images while examining model calibration, visual explanations, and sensitivity to image perturbations.

## Tech Stack

- Python
- PyTorch and Torchvision
- DenseNet-121 transfer learning
- Scikit-learn
- Grad-CAM
- SHAP
- Gradio
- Google Colab

## Dataset

Public pediatric chest X-ray dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset preprocessing included SHA-256 duplicate detection and group-aware splitting.

Exact duplicate images were removed. However, patient independence and near-duplicate leakage were not fully resolved.

## Model

- Architecture: DenseNet-121
- Initialization: ImageNet pretrained weights
- Image size: 224 × 224
- Optimizer: AdamW
- Training epochs: 5
- Best checkpoint: epoch 4
- Model selection: validation ROC-AUC
- Calibration: Platt scaling on validation predictions
- Decision threshold: selected using validation sensitivity

## Evaluation Results

Results on the internal held-out test partition (872 images):

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9963 |
| Sensitivity | 95.43% |
| Specificity | 97.47% |
| Precision | 99.02% |
| Brier score | 0.0161 |
| True positives | 606 |
| True negatives | 231 |
| False positives | 6 |
| False negatives | 29 |

These results do not establish external or clinical generalization.

## Explainability

Grad-CAM visualizes spatial sensitivity of the model's pneumonia logit.

SHAP was used to explore image attribution on selected examples.

Visual explanations do not establish that the model identifies medically meaningful pathology.

## Shortcut-Learning Audit

A paired perturbation experiment compared border occlusion against center occlusion.

| Metric | Border occlusion | Center occlusion |
|---|---:|---:|
| ROC-AUC | 0.9944 | 0.9938 |
| Sensitivity | 96.54% | 75.75% |
| Decision changes | 8 | 144 |
| Mean absolute score change | 0.0297 | 0.0711 |

Central occlusion caused larger prediction changes than border occlusion under the tested perturbations.

This does not prove the absence of shortcut learning.

## Interactive Demo

The Gradio application allows users to:

1. Upload a public, de-identified chest X-ray.
2. Obtain a research-only prediction.
3. View the calibrated model score.
4. Inspect a Grad-CAM visualization.

The application has been tested locally through Google Colab and a temporary Gradio share link.

Permanent hosting has not been established.

## Limitations

- Single pediatric chest X-ray dataset.
- Unresolved patient-independence and near-duplicate concerns.
- No external hospital validation.
- Calibration may not transfer across populations.
- Explanations are not clinically verified.
- Occlusion experiments introduce distribution shifts.
- Not suitable for medical decision-making.

## Repository Structure

- `src/data.py`: dataset preparation
- `src/train.py`: model training
- `src/evaluate.py`: evaluation and calibration
- `src/explain.py`: Grad-CAM and SHAP
- `src/audit.py`: shortcut audit
- `app.py`: Gradio demonstration
- `configs/`: experiment configurations
- `reports/`: evaluation results

## Reproducibility

The original dataset must be downloaded separately.

The trained model checkpoint is not included in this repository by default. The application requires `best_densenet121.pt` and `day3_calibration.json` alongside `app.py`.

Training and evaluation scripts are provided for research reproducibility.

## Responsible Use

This project is an educational machine learning demonstration, not a clinically validated system.

Do not use its predictions to diagnose, exclude, or manage pneumonia.
