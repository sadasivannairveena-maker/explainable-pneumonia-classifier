"""Day 5: frozen-model paired border-versus-center occlusion audit.

Exploratory test-set analysis, not a clinical validation or a causal proof of shortcuts.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import densenet121
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class AuditDataset(Dataset):
    def __init__(self, frame, size, band):
        self.frame = frame.reset_index(drop=True)
        self.size = size
        self.band = band
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        path = self.frame.iloc[idx]['path']
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            x = self.transform(img)
        return x, int(self.frame.iloc[idx]['label'])


def masks(size, band):
    # Border and center occlude approximately equal numbers of pixels.
    width = max(1, round(size * band))
    border = torch.zeros((size, size), dtype=torch.bool)
    border[:width] = True
    border[-width:] = True
    border[:, :width] = True
    border[:, -width:] = True
    border_area = int(border.sum().item())
    side = min(size, round(math.sqrt(border_area)))
    start = (size - side) // 2
    center = torch.zeros_like(border)
    center[start:start+side, start:start+side] = True
    return border, center, width, side


def occlude(batch, mask):
    # In normalized ImageNet coordinates, 0 corresponds to ImageNet mean RGB.
    return batch.masked_fill(mask[None, None, :, :], 0.0)


def predict(model, batch, device, coef, intercept):
    with torch.inference_mode():
        logits = model(batch.to(device)).flatten().float().cpu().numpy()
    z = np.clip(coef * logits + intercept, -80, 80)
    return 1 / (1 + np.exp(-z))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def paired_ci(values, seed=42, n_boot=1000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/content/pneumonia-xai-audit')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--border-fraction', type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.border_fraction < 0.25:
        raise ValueError('border-fraction must be between 0 and 0.25')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    root = Path(args.root)
    manifest = pd.read_csv(root / 'data/manifests/test.csv')
    day3 = pd.read_csv(root / 'reports/day3_test_predictions.csv')
    required = {'path', 'label', 'calibrated_probability', 'predicted_label'}
    if not required.issubset(day3.columns):
        raise ValueError(f'Day 3 predictions missing: {sorted(required - set(day3.columns))}')
    if manifest['path'].duplicated().any() or day3['path'].duplicated().any():
        raise ValueError('Duplicate paths prevent one-to-one alignment')
    if set(manifest.path) != set(day3.path):
        raise ValueError('Manifest and Day 3 prediction paths do not match')
    frame = manifest[['path', 'label']].merge(day3, on='path', suffixes=('', '_day3'), validate='one_to_one')
    if not (frame.label.astype(int) == frame.label_day3.astype(int)).all():
        raise ValueError('Day 3 labels do not match manifest labels')
    cal = json.loads((root / 'artifacts/day3_calibration.json').read_text())
    threshold = float(cal['threshold'])
    coef, intercept = float(cal['coefficient']), float(cal['intercept'])
    ckpt = torch.load(root / 'artifacts/best_densenet121.pt', map_location='cpu', weights_only=False)
    size = int(ckpt.get('image_size', 224))
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    border_mask, center_mask, width, side = masks(size, args.border_fraction)
    loader = DataLoader(AuditDataset(frame, size, args.border_fraction), batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=device.type == 'cuda')
    base, border, center, labels = [], [], [], []
    print('Device:', device, '| images:', len(frame), '| size:', size, flush=True)
    print('Border width:', width, '| central square side:', side, flush=True)
    for idx, (x, y) in enumerate(loader):
        base.extend(predict(model, x, device, coef, intercept))
        border.extend(predict(model, occlude(x, border_mask), device, coef, intercept))
        center.extend(predict(model, occlude(x, center_mask), device, coef, intercept))
        labels.extend(y.numpy().tolist())
        if (idx + 1) % 10 == 0:
            print(f'Processed {min((idx+1)*args.batch_size,len(frame))}/{len(frame)}', flush=True)
    base, border, center, y = map(np.asarray, (base, border, center, labels))
    saved = frame.calibrated_probability.to_numpy(dtype=float)
    maxdiff = float(np.max(np.abs(base - saved)))
    # Numerical differences can arise from CUDA kernels, but large ones indicate wrong setup.
    if maxdiff > 0.01:
        raise RuntimeError(f'Baseline probabilities differ from frozen Day 3 predictions (max {maxdiff:.4f}). Stop: do not audit a mismatched model.')
    if not np.array_equal((base >= threshold).astype(int), frame.predicted_label.to_numpy(dtype=int)):
        raise RuntimeError('Baseline decisions differ from Day 3 predictions')
    out = pd.DataFrame({
        'path': frame.path, 'label': y, 'baseline_probability': base,
        'border_probability': border, 'center_probability': center,
        'baseline_prediction': (base >= threshold).astype(int),
        'border_prediction': (border >= threshold).astype(int),
        'center_prediction': (center >= threshold).astype(int),
        'border_absolute_change': np.abs(border-base),
        'center_absolute_change': np.abs(center-base),
        'border_signed_change': border-base,
        'center_signed_change': center-base,
    })
    out['border_flip'] = out.baseline_prediction != out.border_prediction
    out['center_flip'] = out.baseline_prediction != out.center_prediction
    report = root / 'reports'
    figures = report / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    out.to_csv(report / 'day5_perturbation_predictions.csv', index=False)
    metrics = {}
    for name, p in [('baseline', base), ('border', border), ('center', center)]:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        metrics[name] = {
            'roc_auc': safe_auc(y, p), 'sensitivity': float(tp / (tp+fn)),
            'specificity': float(tn / (tn+fp)), 'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp),
            'mean_absolute_probability_change': float(np.abs(p-base).mean()),
            'decision_flips': int(np.sum(pred != (base >= threshold))),
        }
    diff = np.abs(border-base) - np.abs(center-base)
    result = {
        'n': len(y), 'checkpoint_epoch': int(ckpt['epoch']),
        'threshold': threshold, 'border_fraction': args.border_fraction,
        'border_pixels': int(border_mask.sum()), 'center_pixels': int(center_mask.sum()),
        'mask_fill': 'ImageNet mean RGB, represented as zero after normalization',
        'metrics': metrics,
        'mean_border_minus_center_absolute_change': float(diff.mean()),
        'paired_bootstrap_95pct_ci': paired_ci(diff),
        'baseline_day3_max_absolute_probability_difference': maxdiff,
        'interpretation_limits': [
            'Occlusion creates out-of-distribution inputs and can affect predictions for reasons other than shortcuts.',
            'A central square is not a lung segmentation mask; border and center are only approximate location controls.',
            'Averaging absolute changes does not reveal whether changes help or harm the prediction.',
            'Test set reused for exploratory audit after Day 3 evaluation; do not tune model based on audit outcomes.',
            'Patient-level independence and external clinical validity remain unverified.',
        ],
    }
    (report / 'day5_audit_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([np.abs(border-base), np.abs(center-base)], tick_labels=['Border', 'Center'], showfliers=False)
    axes[0].set(ylabel='Absolute change in calibrated probability', title='Paired perturbation sensitivity')
    names = ['Baseline', 'Border', 'Center']
    axes[1].bar(names, [metrics[k]['sensitivity'] for k in ('baseline','border','center')], label='Sensitivity')
    axes[1].plot(names, [metrics[k]['specificity'] for k in ('baseline','border','center')], marker='o', label='Specificity')
    axes[1].set(ylabel='Fraction', ylim=(0, 1.05), title='Fixed-threshold performance')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / 'day5_perturbation_audit.png', dpi=160)
    plt.close(fig)
    print('\nDAY 5 AUDIT COMPLETE')
    for name, stats in metrics.items():
        print(f"{name}: AUC={stats['roc_auc']:.4f} sensitivity={stats['sensitivity']:.4f} specificity={stats['specificity']:.4f} flips={stats['decision_flips']} mean_abs_change={stats['mean_absolute_probability_change']:.4f}")
    print('Border minus center mean absolute change:', round(float(diff.mean()), 5))
    print('Paired bootstrap 95% CI:', result['paired_bootstrap_95pct_ci'])
    print('Saved:', report / 'day5_audit_summary.json')
    print('Saved:', figures / 'day5_perturbation_audit.png')

if __name__ == '__main__':
    main()
