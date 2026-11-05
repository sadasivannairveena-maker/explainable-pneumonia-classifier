from __future__ import annotations
import argparse, json, random
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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class GradCAM:
    def __init__(self, model):
        self.activations = None
        self.gradients = None
        layer = model.features.denseblock4
        self.forward_handle = layer.register_forward_hook(self._forward)
        self.backward_handle = None
    def _forward(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))
    def close(self):
        self.forward_handle.remove()
        if self.backward_handle is not None: self.backward_handle.remove()

def load_image(path, size):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert('RGB')
        img = img.resize((size,size), Image.Resampling.BILINEAR)
        arr = np.asarray(img).copy()
        tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(img)
    return arr, tensor

def cam_for(model, cam, tensor, device):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logit = model(tensor.unsqueeze(0).to(device)).flatten()[0]
        logit.backward()
    acts, grads = cam.activations, cam.gradients
    weights = grads.mean(dim=(2,3), keepdim=True)
    heat = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    heat = torch.nn.functional.interpolate(heat, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
    heat = heat[0,0].detach().cpu().numpy()
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    return heat, float(torch.sigmoid(logit.detach()).item())

def choose_examples(df, predictions, n_each):
    df = df.copy()
    df['probability'] = predictions
    df['predicted_label'] = (df['probability'] >= 0.5).astype(int)
    rng = np.random.default_rng(42)
    chosen = []
    for label, pred, tag in [(0,0,'normal_correct'),(1,1,'pneumonia_correct'),(0,1,'false_positive'),(1,0,'false_negative')]:
        pool = df[(df.label == label) & (df.predicted_label == pred)]
        if len(pool):
            ids = rng.choice(pool.index.to_numpy(), size=min(n_each,len(pool)), replace=False)
            for i in ids:
                chosen.append((df.loc[i],tag))
    return chosen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/content/pneumonia-xai-audit')
    parser.add_argument('--n-per-category', type=int, default=2)
    parser.add_argument('--shap-images', type=int, default=2)
    parser.add_argument('--shap-background', type=int, default=8)
    parser.add_argument('--shap-samples', type=int, default=40)
    args = parser.parse_args()
    root = Path(args.root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,flush=True)
    checkpoint_path = root/'artifacts/best_densenet121.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    size = int(checkpoint.get('image_size',224))
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features,1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    val_df = pd.read_csv(root/'data/manifests/val.csv')
    test_df = pd.read_csv(root/'data/manifests/test.csv')
    for split, df in [('val',val_df),('test',test_df)]:
        required = {'path','label'}
        if not required.issubset(df.columns):
            raise ValueError(f'{split} CSV needs columns {required}; found {list(df.columns)}')
        if not df.label.isin([0,1]).all():
            raise ValueError(f'{split} labels must be numeric 0/1')
    # Use validation examples to establish the explanation workflow.
    # Test examples are used only for post-hoc error analysis, not model or threshold selection.
    pred_file = root/'reports/day3_test_predictions.csv'
    if not pred_file.exists():
        raise FileNotFoundError(f'Missing Day 3 predictions: {pred_file}')
    preds = pd.read_csv(pred_file)
    print('Day 3 prediction columns:',list(preds.columns),flush=True)
    # Recompute uncalibrated scores in manifest order to avoid assumptions about CSV column names/order.
    # For error categories use Day 3 calibrated decision if its prediction file exposes a suitable column.
    from torch.utils.data import Dataset, DataLoader
    class EvalDataset(Dataset):
        def __len__(self): return len(test_df)
        def __getitem__(self,i): return load_image(test_df.iloc[i].path,size)[1]
    loader = DataLoader(EvalDataset(),batch_size=16,shuffle=False,num_workers=0)
    raw_probs = []
    with torch.no_grad():
        for x in loader:
            raw_probs.extend(torch.sigmoid(model(x.to(device)).flatten()).cpu().numpy().tolist())
    # Use uncalibrated 0.5 categories only to sample examples; label them clearly.
    chosen = choose_examples(test_df,np.asarray(raw_probs),args.n_per_category)
    print('Selected examples:',len(chosen),flush=True)
    out = root/'reports/figures/day4'
    out.mkdir(parents=True,exist_ok=True)
    cam = GradCAM(model)
    rows = []
    selected_tensors = []
    try:
        for idx,(row,category) in enumerate(chosen):
            rgb,tensor = load_image(row.path,size)
            heat,prob = cam_for(model,cam,tensor,device)
            fig,axs = plt.subplots(1,3,figsize=(11,3.7))
            axs[0].imshow(rgb); axs[0].set_title('Original X-ray')
            axs[1].imshow(heat,cmap='jet',vmin=0,vmax=1); axs[1].set_title('Grad-CAM')
            axs[2].imshow(rgb); axs[2].imshow(heat,cmap='jet',alpha=0.4,vmin=0,vmax=1); axs[2].set_title('Overlay')
            for ax in axs: ax.axis('off')
            fig.suptitle(f'{category} | label={int(row.label)} | raw P(pneumonia)={prob:.3f}')
            fig.tight_layout()
            path = out/f'gradcam_{idx:02d}_{category}.png'
            fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
            rows.append({'path':row.path,'true_label':int(row.label),'raw_probability':prob,'sampling_category_raw_0_5':category,'gradcam_figure':str(path)})
            selected_tensors.append((row,category,tensor))
    finally:
        cam.close()
    pd.DataFrame(rows).to_csv(root/'reports/day4_explanations.csv',index=False)
    print('Grad-CAM images saved:',len(rows),flush=True)
    if args.shap_images > 0:
        try:
            import shap
        except ImportError as e:
            raise RuntimeError('SHAP is missing. Install with: %pip install shap==0.49.1') from e
        # GradientExplainer needs gradient-enabled inference; use model logits.
        # A small validation background approximates the reference distribution.
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(val_df),size=min(args.shap_background,len(val_df)),replace=False)
        background = torch.stack([load_image(val_df.iloc[int(i)].path,size)[1] for i in bg_idx]).to(device)
        explainer = shap.GradientExplainer(model,background)
        for idx,(row,category,tensor) in enumerate(selected_tensors[:args.shap_images]):
            values = explainer.shap_values(tensor.unsqueeze(0).to(device),nsamples=args.shap_samples)
            if isinstance(values,list): values=values[0]
            values = np.asarray(values)
            # SHAP versions return [N,C,H,W] or [N,C,H,W,1] for one output.
            values = np.squeeze(values)
            if values.shape != (3,size,size):
                raise RuntimeError(f'Unexpected SHAP shape: {values.shape}')
            signed = values.sum(axis=0)
            bound = float(np.quantile(np.abs(signed),0.99)) or 1e-8
            rgb,_ = load_image(row.path,size)
            fig,axs = plt.subplots(1,2,figsize=(8,4))
            axs[0].imshow(rgb);axs[0].set_title('Original X-ray')
            axs[1].imshow(rgb);axs[1].imshow(signed,cmap='bwr',vmin=-bound,vmax=bound,alpha=0.65)
            axs[1].set_title('SHAP signed attribution (logit)')
            for ax in axs: ax.axis('off')
            fig.suptitle(f'{category} | red increases pneumonia logit, blue decreases')
            fig.tight_layout()
            path = out/f'shap_{idx:02d}_{category}.png'
            fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
            print('Saved SHAP:',path.name,flush=True)
    summary = {'checkpoint_epoch':int(checkpoint['epoch']),'device':str(device),'gradcam_images':len(rows),'shap_images':min(args.shap_images,len(rows)),'explanation_selection':'test images sampled by raw score 0.5; NOT calibrated Day 3 decision','notes':'Post-hoc visualizations; not evidence of clinical causality or patient-independent generalization'}
    (root/'reports/day4_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nDAY 4 COMPLETE')
    print(json.dumps(summary,indent=2))
    print('Figures:',out)

if __name__=='__main__': main()
