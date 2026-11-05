"""Day 3: validation-only calibration/threshold selection, frozen test evaluation."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss,
                             log_loss, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121

class ImageManifest(Dataset):
    def __init__(self, path, image_size):
        self.df = pd.read_csv(path)
        for col in ('path', 'label', 'sha256', 'group_key'):
            if col not in self.df: raise ValueError(f'Missing column {col} in {path}')
        if not self.df['label'].isin([0, 1]).all():
            raise ValueError(f'Expected numeric labels 0/1 in {path}')
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(str(row['path'])) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            image = self.transform(img)
        return image, int(row['label'])

def infer(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == 'cuda'))
    scores, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device)).flatten()
            scores.extend(logits.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    return np.array(labels, dtype=int), np.array(scores, dtype=float)

def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -80, 80)))

def select_threshold(y, probs, target=.95):
    # Highest threshold satisfying sensitivity target on validation data.
    # Include 0 to guarantee a feasible candidate; ties handled deterministically.
    thresholds = np.unique(np.concatenate(([0.], probs)))
    selected = 0.
    for threshold in thresholds:
        sensitivity = np.mean(probs[y == 1] >= threshold)
        if sensitivity >= target: selected = float(threshold)
    return selected

def metrics(y, probs, threshold):
    predicted = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), positives=int(sum(y)), negatives=int(len(y)-sum(y)),
                roc_auc=float(roc_auc_score(y, probs)),
                average_precision=float(average_precision_score(y, probs)),
                brier_score=float(brier_score_loss(y, probs)),
                log_loss=float(log_loss(y, np.clip(probs, 1e-7, 1-1e-7))),
                threshold=float(threshold), accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                sensitivity=float(recall_score(y, predicted, pos_label=1)),
                specificity=float(tn/(tn+fp)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def save_plots(y, probs, threshold, out):
    fpr, tpr, _ = roc_curve(y, probs)
    precision, recall, _ = precision_recall_curve(y, probs)
    tn, fp, fn, tp = confusion_matrix(y, probs >= threshold, labels=[0,1]).ravel()
    cm = np.array([[tn, fp],[fn,tp]])
    fig, axes = plt.subplots(2,2, figsize=(11,9))
    axes[0,0].plot(fpr,tpr, label=f'AUC={roc_auc_score(y,probs):.4f}')
    axes[0,0].plot([0,1],[0,1],'--',color='gray')
    axes[0,0].set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC')
    axes[0,0].legend()
    axes[0,1].plot(recall,precision, label=f'AP={average_precision_score(y,probs):.4f}')
    axes[0,1].set(xlabel='Recall',ylabel='Precision',title='Test precision–recall')
    axes[0,1].legend()
    axes[1,0].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1,0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,
                           color='white' if cm[i,j] > cm.max()/2 else 'black')
    axes[1,0].set(xticks=[0,1],yticks=[0,1],xticklabels=['NORMAL','PNEUMONIA'],
                  yticklabels=['NORMAL','PNEUMONIA'],xlabel='Predicted',ylabel='Actual',
                  title='Test confusion matrix')
    frac, avg = calibration_curve(y, probs, n_bins=10, strategy='quantile')
    axes[1,1].plot([0,1],[0,1],'--',color='gray',label='Perfect')
    axes[1,1].plot(avg,frac,'o-',label='Calibrated model')
    axes[1,1].set(xlabel='Mean predicted probability',ylabel='Observed fraction pneumonia',
                  title='Test reliability diagram',xlim=(0,1),ylim=(0,1))
    axes[1,1].legend()
    fig.tight_layout()
    fig.savefig(out,dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='/content/pneumonia-xai-audit')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--sensitivity-target',type=float,default=.95)
    args = parser.parse_args()
    if not 0 < args.sensitivity_target <= 1: raise ValueError('Sensitivity target must be in (0,1]')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root = Path(args.root)
    manifest = root/'data/manifests'
    checkpoint_path = root/'artifacts/best_densenet121.pt'
    for path in [manifest/'train.csv',manifest/'val.csv',manifest/'test.csv',checkpoint_path]:
        if not path.exists(): raise FileNotFoundError(f'Missing required file: {path}')
    frames = {name: pd.read_csv(manifest/f'{name}.csv') for name in ('train','val','test')}
    for i, a in enumerate(('train','val','test')):
        for b in ('train','val','test')[i+1:]:
            for col in ('sha256','group_key'):
                if set(frames[a][col]) & set(frames[b][col]):
                    raise RuntimeError(f'{col} overlap between {a} and {b}')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint.get('model_name') != 'densenet121': raise ValueError('Unexpected model checkpoint')
    image_size = int(checkpoint.get('image_size',224))
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features,1)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print('Device:',device,'| Checkpoint epoch:',checkpoint.get('epoch'),flush=True)
    val_ds = ImageManifest(manifest/'val.csv',image_size)
    test_ds = ImageManifest(manifest/'test.csv',image_size)
    print('Running validation inference...',flush=True)
    y_val, logits_val = infer(model,val_ds,device,args.batch_size)
    # Fit Platt scaling on validation only. Model scores are raw logits.
    calibrator = LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=42)
    calibrator.fit(logits_val.reshape(-1,1),y_val)
    val_raw = sigmoid(logits_val)
    val_cal = calibrator.predict_proba(logits_val.reshape(-1,1))[:,1]
    threshold = select_threshold(y_val,val_cal,args.sensitivity_target)
    print('Validation threshold:',round(threshold,6),flush=True)
    print('Validation calibrated sensitivity:',round(np.mean(val_cal[y_val==1]>=threshold),4),flush=True)
    # Freeze calibration and threshold before test inference.
    print('Running untouched test inference...',flush=True)
    y_test, logits_test = infer(model,test_ds,device,args.batch_size)
    test_raw = sigmoid(logits_test)
    test_cal = calibrator.predict_proba(logits_test.reshape(-1,1))[:,1]
    results = {
        'checkpoint_epoch':int(checkpoint.get('epoch',-1)),
        'image_size':image_size,'validation_target_sensitivity':args.sensitivity_target,
        'selected_threshold':threshold,
        'platt_coef':float(calibrator.coef_[0,0]),
        'platt_intercept':float(calibrator.intercept_[0]),
        'validation_raw':metrics(y_val,val_raw,.5),
        'validation_calibrated':metrics(y_val,val_cal,threshold),
        'test_raw_at_0_5':metrics(y_test,test_raw,.5),
        'test_calibrated':metrics(y_test,test_cal,threshold),
        'limitations':[
            'Calibration and threshold both selected on validation partition.',
            'Patient-level independence not established from available filenames.',
            'Near-duplicate visual candidates not exhaustively adjudicated.',
            'Dataset-specific shortcut learning not yet audited.',
            'Not validated for clinical use.'
        ]
    }
    report = root/'reports'; report.mkdir(parents=True,exist_ok=True)
    figures = report/'figures'; figures.mkdir(parents=True,exist_ok=True)
    (report/'day3_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    pd.DataFrame({'path':val_ds.df['path'],'label':y_val,'logit':logits_val,
                  'raw_probability':val_raw,'calibrated_probability':val_cal,
                  'predicted_label':(val_cal>=threshold).astype(int)}).to_csv(
                  report/'day3_validation_predictions.csv',index=False)
    pd.DataFrame({'path':test_ds.df['path'],'label':y_test,'logit':logits_test,
                  'raw_probability':test_raw,'calibrated_probability':test_cal,
                  'predicted_label':(test_cal>=threshold).astype(int)}).to_csv(
                  report/'day3_test_predictions.csv',index=False)
    save_plots(y_test,test_cal,threshold,figures/'day3_test_evaluation.png')
    (root/'artifacts/day3_calibration.json').write_text(json.dumps({
        'method':'Platt logistic regression on raw DenseNet logits',
        'coefficient':float(calibrator.coef_[0,0]),
        'intercept':float(calibrator.intercept_[0]),
        'threshold':threshold,'target_validation_sensitivity':args.sensitivity_target,
        'positive_class':'PNEUMONIA',
    },indent=2),encoding='utf-8')
    print('\nDAY 3 COMPLETE')
    print('Test ROC-AUC:',round(results['test_calibrated']['roc_auc'],4))
    print('Test sensitivity:',round(results['test_calibrated']['sensitivity'],4))
    print('Test specificity:',round(results['test_calibrated']['specificity'],4))
    print('Test precision:',round(results['test_calibrated']['precision'],4))
    print('Test Brier score:',round(results['test_calibrated']['brier_score'],4))
    print('Confusion matrix TN FP FN TP:',*[results['test_calibrated'][k] for k in ('tn','fp','fn','tp')])
    print('Saved:',report/'day3_metrics.json')
    print('Saved:',figures/'day3_test_evaluation.png')

if __name__ == '__main__': main()
