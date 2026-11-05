# Model Card — DenseNet-121 Pneumonia Classifier

## Model Details

**Architecture:** DenseNet-121 with a binary classification head.

**Framework:** PyTorch / Torchvision.

**Task:** Binary classification of pediatric chest X-ray images into NORMAL and PNEUMONIA dataset labels.

**Training:** ImageNet-pretrained transfer learning, five epochs, with the best checkpoint selected at epoch four.

## Intended Use

Educational research, machine learning portfolio demonstration, and exploration of explainability and robustness techniques.

## Out-of-Scope Use

- Clinical diagnosis
- Medical triage
- Patient risk assessment
- Autonomous healthcare decisions
- Generalization to adult populations without validation

## Dataset and Evaluation

The model was developed using the public pediatric chest X-ray dataset distributed on Kaggle.

Internal held-out test partition: 872 images.

ROC-AUC: 0.9963

Sensitivity: 0.9543

Specificity: 0.9747

Precision: 0.9902

Brier score: 0.0161

Confusion matrix:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 231 | 6 |
| Actual Pneumonia | 29 | 606 |

## Calibration

Platt scaling was fitted using the validation partition.

The frozen decision threshold was approximately 0.899805, selected to target at least 95% validation sensitivity.

The calibrated score should not be interpreted as a clinically validated disease probability.

## Explainability

Grad-CAM and SHAP provide exploratory model-attribution visualizations.

These methods do not confirm that highlighted image regions correspond to disease findings.

## Shortcut Audit

Border and center occlusion were evaluated using a frozen model and calibration pipeline.

Mean absolute calibrated probability change:

- Border: 0.0297
- Center: 0.0711

The model was more sensitive to center occlusion in this experiment.

This does not establish freedom from shortcut learning.

## Known Limitations

- Pediatric-only dataset
- Potential selection and acquisition biases
- Patient independence not fully verified
- Visually similar-image leakage not fully ruled out
- No external validation
- Possible dataset-specific calibration
- Artificial perturbations may create out-of-distribution inputs

## Ethical and Safety Considerations

Research-only demonstration.

Do not upload identifiable medical images to publicly hosted demos.

The model must not be used for clinical decisions.
