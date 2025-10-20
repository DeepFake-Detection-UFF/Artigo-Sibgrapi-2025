import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import slic, mark_boundaries
from scipy.stats import entropy

# ==================== CONFIGURAÇÕES ====================
DATASET_ROOT = r"E:\Downloads\faceforensis dataset archive\FaceForensics++_C23\output_frames\split_dataset\val"
MODEL_PATH = "./models_resnet/resnet50_multiclass.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega as classes com base na estrutura real das pastas
dataset_temp = ImageFolder(DATASET_ROOT)
class_names = dataset_temp.classes
print("Classes carregadas:", class_names)

# ==================== MODELO ====================
def load_resnet50_multiclass(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==================== FUNÇÕES AUXILIARES ====================
def make_gradcam_pytorch(model, img_tensor, target_layer='layer4'):
    gradients, activations = [], []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break
    output = model(img_tensor)
    class_idx = torch.argmax(output)
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()
    grad = gradients[0]
    act = activations[0]
    pooled_grad = torch.mean(grad, dim=[0, 2, 3])
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_grad[i]
    heatmap = act.mean(dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-12)
    return heatmap

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    image_np = np.array(image.resize((224, 224))) / 255.0
    return image_tensor, image_np

def salience_entropy(smap):
    smap = smap / (np.sum(smap) + 1e-12)
    return entropy(smap.flatten() + 1e-12)

def reaction_to_noise(model, img_tensor, original_salmap):
    noisy = img_tensor + torch.normal(0, 0.05, img_tensor.shape).to(DEVICE)
    noisy_salmap = make_gradcam_pytorch(model, noisy)
    return np.mean(np.abs(original_salmap - noisy_salmap))

def geometrical_robustness(model, raw_img, original_salmap):
    flipped = np.flip(raw_img, axis=1).copy()
    flipped_tensor = torch.tensor(flipped.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    flipped_salmap = make_gradcam_pytorch(model, flipped_tensor)
    return np.mean(np.abs(original_salmap - flipped_salmap))

def accuracy_over_segmentation(model, image, saliency_map, segments=50):
    salmap_resized = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    img_uint8 = (image * 255).astype(np.uint8)
    segs = slic(img_uint8, n_segments=segments, compactness=20, enforce_connectivity=True)
    relevance = np.array([np.mean(salmap_resized[segs == i]) for i in range(np.max(segs) + 1)])
    top_k = np.argsort(relevance)[-int(segments * 0.2):]
    mask = np.isin(segs, top_k).astype(np.float32)
    if mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    degraded = image * (1 - mask)
    with torch.no_grad():
        orig_pred = model(torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
        degr_pred = model(torch.tensor(degraded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
    return np.abs(orig_pred.cpu().numpy() - degr_pred.cpu().numpy()).mean(), segs

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    model = load_resnet50_multiclass(MODEL_PATH, len(class_names))

    image_files = []
    for folder in sorted(os.listdir(DATASET_ROOT)):
        folder_path = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(folder_path):
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
            if files:
                image_files.append(os.path.join(folder_path, files[0]))  # ou todas se quiser: image_files.extend(...)

    if not image_files:
        print("Nenhuma imagem encontrada nas subpastas!")
        exit()

    fig, axs = plt.subplots(nrows=len(image_files), ncols=3, figsize=(20, 5 * len(image_files)),
                            gridspec_kw={'hspace': 0.5, 'wspace': 0.25})

    for i, path in enumerate(image_files):
        img_tensor, raw_img = preprocess_image(path)
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        salmap = make_gradcam_pytorch(model, img_tensor)
        entropy_val = salience_entropy(salmap)
        noise_val = reaction_to_noise(model, img_tensor, salmap)
        geo_val = geometrical_robustness(model, raw_img, salmap)
        aos_val, segs = accuracy_over_segmentation(model, raw_img, salmap)

        heatmap_resized = cv2.resize(salmap, (raw_img.shape[1], raw_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) / 255.0
        overlay = 0.6 * raw_img + 0.4 * heatmap_color

        top3_idx = probs.argsort()[-3:][::-1]
        top3_str = "\n".join([f"{class_names[idx]} ({probs[idx]*100:.1f}%)" for idx in top3_idx])
        pred_label = class_names[pred_idx]
        true_label = os.path.basename(os.path.dirname(path))

        axs[i, 0].imshow(raw_img)
        axs[i, 0].set_title(f"Original: {true_label}\nClasse prevista: {pred_label}", fontsize=16)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(mark_boundaries(raw_img, segs, color=(1, 1, 0)))
        axs[i, 1].set_title("Segmentação SLIC", fontsize=16)
        axs[i, 1].axis("off")

        axs[i, 2].imshow(overlay)
        axs[i, 2].set_title("Grad-CAM + Métricas + Classificação", fontsize=16)
        axs[i, 2].annotate(
            f"Entropia: {entropy_val:.4f}\nRuído: {noise_val:.4f}\nGeo: {geo_val:.4f}\nAOS: {aos_val:.4f}",
            xy=(1.0, 0.0), xycoords='axes fraction',
            xytext=(10, 0), textcoords='offset points',
            fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        axs[i, 2].annotate(
            f"Top-3:\n{top3_str}",
            xy=(1.0, 0.0), xycoords='axes fraction',
            xytext=(-350, 0), textcoords='offset points',
            fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_result.png", bbox_inches='tight', dpi=150)
    plt.show()
