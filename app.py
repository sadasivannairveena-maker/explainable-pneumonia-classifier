"""Research-only DenseNet121 chest X-ray explorer. No clinical use."""
from __future__ import annotations
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'best_densenet121.pt'
CALIBRATION = ROOT / 'day3_calibration.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not CHECKPOINT.is_file():
    raise FileNotFoundError('Missing best_densenet121.pt beside app.py')
if not CALIBRATION.is_file():
    raise FileNotFoundError('Missing day3_calibration.json beside app.py')

# Checkpoint is created by our own trusted training code; never load untrusted .pt files.
try:
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
except Exception as exc:
    raise RuntimeError('Could not safely load the trusted model checkpoint.') from exc
if checkpoint.get('model_name') != 'densenet121':
    raise ValueError('Unexpected checkpoint architecture')
IMAGE_SIZE = int(checkpoint.get('image_size', 224))
model = densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(DEVICE).eval()

with CALIBRATION.open(encoding='utf-8') as f:
    calibration = json.load(f)
if calibration.get('positive_class') != 'PNEUMONIA':
    raise ValueError('Calibration positive class mismatch')
COEF = float(calibration['coefficient'])
INTERCEPT = float(calibration['intercept'])
THRESHOLD = float(calibration['threshold'])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Hook the last dense block. Registering on the block output makes the
# explanation consistent with the original Day 4 implementation.
class GradCAM:
    def __init__(self, network):
        self.activations = None
        self.gradients = None
        self.handle = network.features.denseblock4.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._grad_hook)

    def _grad_hook(self, gradient):
        self.gradients = gradient

    def compute(self, tensor):
        self.activations = None
        self.gradients = None
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = model(tensor).reshape(-1)[0]
            logit.backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError('Grad-CAM hooks did not capture gradients')
        weights = self.gradients.detach().float().mean(dim=(2, 3), keepdim=True)
        heat = torch.relu((weights * self.activations.detach().float()).sum(dim=1))[0]
        heat = heat.cpu().numpy()
        heat -= heat.min()
        heat /= (heat.max() + 1e-8)
        return float(logit.detach().cpu()), heat

cam = GradCAM(model)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))


def make_overlay(image, heat):
    import matplotlib
    from matplotlib import colormaps
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    base = np.asarray(resized.convert('RGB'), dtype=np.float32) / 255.0
    h = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255), mode='L')
    h = h.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    normalized = np.asarray(h, dtype=np.float32) / 255.0
    colored = colormaps['jet'](normalized)[..., :3]
    alpha = (0.42 * normalized)[..., None]
    result = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(np.clip(result * 255, 0, 255)))


def predict(image):
    if image is None:
        raise gr.Error('Upload a chest X-ray image first.')
    if not isinstance(image, Image.Image):
        raise gr.Error('Invalid image input.')
    if image.width < 64 or image.height < 64:
        raise gr.Error('Image is too small; use at least 64×64 pixels.')
    image = ImageOps.exif_transpose(image).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    logit, heat = cam.compute(tensor)
    calibrated = float(sigmoid(COEF * logit + INTERCEPT))
    predicted = 'PNEUMONIA' if calibrated >= THRESHOLD else 'NORMAL'
    overlay = make_overlay(image, heat)
    result = (
        f'### Model output: {predicted}\n'
        f'**Calibrated pneumonia score:** {calibrated:.4f} ({calibrated * 100:.2f}%)  \n'
        f'**Frozen decision threshold:** {THRESHOLD:.6f}  \n'
        f'**Raw model score:** {logit:.4f}  \n\n'
        '**Research-only result.** The score is calibrated on the project validation '
        'partition, not a hospital population. A heatmap is not a diagnosis or proof '
        'of medically meaningful attention.'
    )
    return overlay, result

with gr.Blocks(title='Chest X-ray AI Explorer') as demo:
    gr.Markdown('''# Chest X-ray AI Explorer
**Research demonstration — NOT for clinical use.** This tool is not a medical device,
not validated for diagnosis, and must not guide medical decisions. Do not upload
patient-identifiable images or protected health information. Use public, de-identified
research images only. Uploaded images are processed by the hosting service.''')
    with gr.Row():
        input_image = gr.Image(type='pil', label='Upload a de-identified frontal chest X-ray')
        output_image = gr.Image(type='pil', label='Grad-CAM overlay (positive pneumonia logit)')
    analyze = gr.Button('Analyze research image', variant='primary')
    result = gr.Markdown()
    analyze.click(fn=predict, inputs=input_image, outputs=[output_image, result], concurrency_limit=1)
    gr.Markdown('''### Limitations
- Trained on a single public pediatric chest X-ray dataset; external generalization unknown.
- Patient independence and visually similar-image leakage were not fully resolved.
- The validation-selected threshold and calibration may not transfer to new populations.
- Grad-CAM highlights sensitivity of a model score, not verified pathology.
- Performance on project test partition: ROC-AUC 0.9963, sensitivity 95.43%, specificity 97.47%; these are **not clinical performance guarantees**.
''')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')))
