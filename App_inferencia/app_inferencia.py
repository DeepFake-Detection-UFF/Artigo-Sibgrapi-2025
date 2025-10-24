# app.py
# Deepfake Forensics – Classificação + XAI (vídeo -> frames)
# UI Streamlit + Fallback CLI, com SLIC (segmentação), métricas e classificação
# e exibição dos nomes das classes detectadas.
#
# Requisitos:
#   pip install streamlit torch torchvision pillow opencv-python-headless scikit-image scipy matplotlib pandas
#
# UI:
#   streamlit run app.py
#
# CLI (exemplo):
#   python app.py --model "C:\MODELS\meu_modelo.pth" --video "C:\VIDEOS\video.mp4" --frames 32 --segment_face --outdir "C:\MODELS_OUTPUTS"

import os
import cv2
import time
import glob
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic, mark_boundaries
from scipy.stats import entropy

# ===================== CONFIG PADRÃO =====================
DEFAULT_MODELS_DIR = r"E:\MODELS"
DEFAULT_VIDEOS_DIR = r"E:\VIDEOS"
DEFAULT_OUTPUT_DIR = r"E:\MODELS_OUTPUTS"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== DETECTOR DE CONTEXTO STREAMLIT =====================
def _is_streamlit_context():
    try:
        from streamlit.runtime.scriptrun_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        try:
            import streamlit as st  # noqa
            return getattr(st, "_is_running_with_streamlit", False)
        except Exception:
            return False

# ===================== PRÉ-PROCESSAMENTO =====================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def pil_from_bgr(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def tensor_from_pil(img_pil):
    return preprocess(img_pil).unsqueeze(0).to(DEVICE)

# ===================== CARREGAMENTO DO MODELO =====================
def infer_num_classes_from_state_dict(sd: dict) -> int:
    # tenta inferir a partir de camadas finais usuais
    for k in ["fc.weight", "classifier.weight", "head.weight"]:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 2:
            return sd[k].shape[0]
    # fallback: maior matriz 2D
    out = None
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            out = v.shape[0]
    if out is None:
        raise RuntimeError("Não foi possível inferir num_classes a partir do checkpoint.")
    return out

def build_resnet50_for_checkpoint(sd: dict):
    num_classes = infer_num_classes_from_state_dict(sd)
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model, num_classes

def load_model_auto(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = {k.replace("model.", "").replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    elif isinstance(ckpt, dict):
        sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
    else:
        sd = ckpt
    model, num_classes = build_resnet50_for_checkpoint(sd)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, num_classes

def maybe_load_class_names(model_path, num_classes):
    """
    Tenta ler lista de classes (um nome por linha) ou usa defaults conhecidos (FaceForensics++).
    Se existir <modelo>.classes.txt ou classes.txt ao lado do .pth, usa o arquivo.
    """
    base = os.path.splitext(model_path)[0]
    candidates = [
        base + ".classes.txt",
        os.path.join(os.path.dirname(model_path), "classes.txt")
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # aceita qualquer quantidade >= 2; se maior que num_classes, corta
            if len(lines) >= 2:
                return lines[:num_classes]

    # fallback padrão para deepfake forensics (FaceForensics++ e correlatos)
    default_classes = [
        "DeepFakeDetection",
        "FaceShifter",
        "FaceSwap",
        "NeuraTextures",
        "deepfakes",
        "face2face",
        "original"
    ]
    # garante tamanho
    if num_classes <= len(default_classes):
        return default_classes[:num_classes]
    # se o modelo tiver mais saídas, completa com nomes genéricos
    return default_classes + [f"class_{i}" for i in range(len(default_classes), num_classes)]

# ===================== EXPLAINABILITY =====================
def make_gradcam_pytorch(model, img_tensor, target_layer='layer4'):
    gradients, activations = [], []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
    # hooks
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break
    output = model(img_tensor)
    class_idx = torch.argmax(output, dim=1)
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
    return heatmap, output.detach()

def saliency_entropy(smap):
    smap = smap / (np.sum(smap) + 1e-12)
    return float(entropy(smap.flatten() + 1e-12))

def reaction_to_noise(model, img_tensor, original_salmap, target_layer='layer4'):
    noisy = img_tensor + torch.normal(0, 0.05, img_tensor.shape).to(DEVICE)
    noisy_salmap, _ = make_gradcam_pytorch(model, noisy, target_layer=target_layer)
    return float(np.mean(np.abs(original_salmap - noisy_salmap)))

def geometrical_robustness(model, raw_img_rgb_224, original_salmap, target_layer='layer4'):
    flipped = np.flip(raw_img_rgb_224, axis=1).copy()
    flipped_tensor = preprocess(Image.fromarray(flipped)).unsqueeze(0).to(DEVICE)
    flipped_salmap, _ = make_gradcam_pytorch(model, flipped_tensor, target_layer=target_layer)
    return float(np.mean(np.abs(original_salmap - flipped_salmap)))

def accuracy_over_segmentation(model, raw_img_rgb_224, saliency_map, segments=50, compactness=20):
    salmap_resized = cv2.resize(saliency_map, (raw_img_rgb_224.shape[1], raw_img_rgb_224.shape[0]))
    img_uint8 = raw_img_rgb_224.astype(np.uint8)
    segs = slic(img_uint8, n_segments=segments, compactness=compactness, enforce_connectivity=True)
    relevance = np.array([np.mean(salmap_resized[segs == i]) for i in range(np.max(segs) + 1)])
    top_k = np.argsort(relevance)[-max(1, int(segments * 0.2)):]
    mask = np.isin(segs, top_k).astype(np.float32)
    if mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    degraded = (img_uint8 * (1 - mask)).astype(np.uint8)
    with torch.no_grad():
        t_ori = preprocess(Image.fromarray(img_uint8)).unsqueeze(0).to(DEVICE)
        logits_ori = model(t_ori)[0].detach().cpu().numpy()
        t_deg = preprocess(Image.fromarray(degraded)).unsqueeze(0).to(DEVICE)
        logits_deg = model(t_deg)[0].detach().cpu().numpy()
    aos = float(np.abs(logits_ori - logits_deg).mean())
    return aos, segs

def overlay_heatmap_on_rgb(rgb224, heatmap):
    hm = cv2.resize(heatmap, (rgb224.shape[1], rgb224.shape[0]))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * rgb224 + 0.4 * hm_color).astype(np.uint8)
    return overlay

def slic_with_boundaries(rgb224, segments=50, compactness=20):
    img_uint8 = rgb224.astype(np.uint8)
    segs = slic(img_uint8, n_segments=segments, compactness=compactness, enforce_connectivity=True)
    vis = mark_boundaries(img_uint8.astype(np.float32)/255.0, segs, color=(1,1,0))
    vis_rgb = (vis*255).astype(np.uint8)
    return segs, vis_rgb

# ===================== VÍDEO / FRAMES =====================
def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_largest_face_bgr(frame_bgr, scale_factor=1.1, min_neighbors=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_detector().detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if len(faces) == 0:
        return frame_bgr
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    pad = int(0.10 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w + pad)
    y1 = min(frame_bgr.shape[0], y + h + pad)
    return frame_bgr[y0:y1, x0:x1, :]

def sample_frames_uniform(video_path, num_samples):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, length - 1, num=num_samples, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames.append(frame)  # BGR
    cap.release()
    return frames

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ===================== PIPELINE COMUM (UI/CLI) =====================
def process_video(
    model_path, video_path, num_frames, segment_face, outdir,
    target_layer="layer4", save_images=True,
    slic_segments=50, slic_compactness=20,
    max_preview=6, class_names=None, ui_callbacks=None
):
    """
    Executa o pipeline em comum para UI e CLI.
    ui_callbacks (opcional): {'on_progress': func(msg), 'on_triplet': func(img_orig, img_slic, img_cam, caption)}
    """
    t0 = time.time()
    ensure_dir(outdir)
    run_name = f"{os.path.splitext(os.path.basename(video_path))[0]}__{os.path.splitext(os.path.basename(model_path))[0]}__{int(time.time())}"
    run_dir = os.path.join(outdir, run_name)
    imgs_dir = os.path.join(run_dir, "frames")
    ensure_dir(imgs_dir)

    if ui_callbacks and ui_callbacks.get("on_progress"):
        ui_callbacks["on_progress"]("Carregando modelo...")

    model, num_classes = load_model_auto(model_path)
    if class_names is None:
        class_names = maybe_load_class_names(model_path, num_classes)

    # >>> mostrar classes detectadas na UI/CLI
    if ui_callbacks and ui_callbacks.get("on_progress"):
        ui_callbacks["on_progress"](f"Modelo carregado. Classes: {num_classes}")
        ui_callbacks["on_progress"](f"Classes detectadas: {class_names}")

    frames_bgr = sample_frames_uniform(video_path, num_frames)
    if not frames_bgr:
        raise RuntimeError("Não foi possível extrair frames do vídeo.")

    results = []
    shown = 0

    for idx, frame_bgr in enumerate(frames_bgr, start=1):
        if segment_face:
            frame_bgr = crop_largest_face_bgr(frame_bgr)

        pil_img = pil_from_bgr(frame_bgr).resize((224, 224))
        rgb224 = np.array(pil_img)  # [0..255]
        img_tensor = tensor_from_pil(pil_img)

        # Grad-CAM + logits
        salmap, logits = make_gradcam_pytorch(model, img_tensor, target_layer=target_layer)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(int(i), float(probs[i]), class_names[int(i)] if int(i) < len(class_names) else f"class_{int(i)}")
                for i in top3_idx]

        # SLIC + visualização
        _, slic_vis = slic_with_boundaries(rgb224, segments=slic_segments, compactness=slic_compactness)

        # Métricas
        ent = saliency_entropy(salmap)
        r_noise = reaction_to_noise(model, img_tensor, salmap, target_layer=target_layer)
        r_geo = geometrical_robustness(model, rgb224, salmap, target_layer=target_layer)
        aos, _ = accuracy_over_segmentation(
            model, rgb224, salmap, segments=slic_segments, compactness=slic_compactness
        )

        # Overlays Grad-CAM
        overlay = overlay_heatmap_on_rgb(rgb224, salmap)

        # Salvar imagens
        orig_path = os.path.join(imgs_dir, f"frame_{idx:04d}_orig.png")
        slic_path = os.path.join(imgs_dir, f"frame_{idx:04d}_slic.png")
        cam_path  = os.path.join(imgs_dir, f"frame_{idx:04d}_gradcam.png")
        if save_images:
            Image.fromarray(rgb224).save(orig_path)
            Image.fromarray(slic_vis).save(slic_path)
            Image.fromarray(overlay).save(cam_path)

        # UI: três painéis
        if ui_callbacks and ui_callbacks.get("on_triplet") and shown < max_preview:
            top3_txt = " | ".join([f"{lbl}:{p*100:.1f}%" for _, p, lbl in top3])
            cap = (f"Frame {idx} | Pred: {class_names[pred_idx] if pred_idx < len(class_names) else pred_idx}  "
                   f"| Ent:{ent:.4f}  Ruído:{r_noise:.4f}  Geo:{r_geo:.4f}  AOS:{aos:.4f}  "
                   f"| Top-3: {top3_txt}")
            ui_callbacks["on_triplet"](rgb224, slic_vis, overlay, cap)
            shown += 1

        results.append({
            "frame_idx": idx,
            "pred_idx": pred_idx,
            "pred_label": class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}",
            "top3": json.dumps(top3, ensure_ascii=False),
            "saliency_entropy": ent,
            "salience_reaction_to_noise": r_noise,
            "salience_resilience_geometric": r_geo,
            "accuracy_over_slic": aos,
            "orig_path": orig_path if save_images else "",
            "slic_path": slic_path if save_images else "",
            "gradcam_path": cam_path if save_images else ""
        })

    df = pd.DataFrame(results)
    summary = df[["saliency_entropy", "salience_reaction_to_noise",
                  "salience_resilience_geometric", "accuracy_over_slic"]].mean().to_frame("mean").T

    ensure_dir(run_dir)
    df_path = os.path.join(run_dir, "per_frame_results.csv")
    summary_path = os.path.join(run_dir, "summary_metrics.csv")
    df.to_csv(df_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    total_time = time.time() - t0
    return {
        "run_dir": run_dir,
        "df": df,
        "summary": summary,
        "time_sec": total_time,
        "num_classes": len(class_names)
    }

# ===================== UI STREAMLIT =====================
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Deepfake Forensics App", layout="wide")
    st.title("🔍 Deepfake Forensics – Classificação + XAI (Original | SLIC | Grad-CAM)")
    st.caption(f"Dispositivo: **{DEVICE}**")

    with st.sidebar:
        st.header("Configurações")
        models_dir = st.text_input("Pasta de modelos (.pth)", DEFAULT_MODELS_DIR)
        videos_dir = st.text_input("Pasta de vídeos", DEFAULT_VIDEOS_DIR)
        output_dir = st.text_input("Pasta de saída (resultados)", DEFAULT_OUTPUT_DIR)

        model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
        if not model_paths:
            st.warning("Nenhum .pth encontrado (ajuste o caminho).")
        model_path = st.selectbox("Escolha o modelo (.pth)", model_paths, index=0 if model_paths else None)

        video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")
        video_files = []
        if os.path.isdir(videos_dir):
            for ext in video_exts:
                video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        video_files = sorted(video_files)
        if not video_files:
            st.warning("Nenhum vídeo encontrado (ajuste o caminho).")
        video_path = st.selectbox("Escolha o vídeo", video_files, index=0 if video_files else None)

        num_frames = st.number_input("Número de frames (amostragem uniforme)", min_value=1, max_value=2000, value=32, step=1)
        segment_face = st.checkbox("Segmentar rosto (HaarCascade)", value=True)
        target_layer = st.text_input("Camada para Grad-CAM (ResNet)", value="layer4")
        slic_segments = st.slider("SLIC - n_segments", min_value=20, max_value=300, value=50, step=5)
        slic_compactness = st.slider("SLIC - compactness", min_value=1, max_value=40, value=20, step=1)
        save_images = st.checkbox("Salvar PNGs dos painéis no disco", value=True)

        run_btn = st.button("▶️ Executar")

    placeholder_progress = st.empty()
    cols = st.columns(3)
    shown = {"n": 0}

    def on_progress(msg):
        placeholder_progress.info(msg)

    def on_triplet(img_orig, img_slic, img_cam, caption):
        i = shown["n"]
        with cols[0]:
            st.image(img_orig, caption="Original (224x224)", use_column_width=True)
        with cols[1]:
            st.image(img_slic, caption="SLIC (contornos)", use_column_width=True)
        with cols[2]:
            st.image(img_cam, caption=caption, use_column_width=True)
        shown["n"] += 1

    if run_btn and model_path and video_path:
        try:
            res = process_video(
                model_path=model_path,
                video_path=video_path,
                num_frames=int(num_frames),
                segment_face=segment_face,
                outdir=output_dir,
                target_layer=target_layer,
                save_images=save_images,
                slic_segments=int(slic_segments),
                slic_compactness=int(slic_compactness),
                ui_callbacks={"on_progress": on_progress, "on_triplet": on_triplet}
            )
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            return

        st.success("Processamento finalizado.")
        st.subheader("📊 Resumo das métricas (média por vídeo)")
        st.dataframe(res["summary"], use_container_width=True)

        st.download_button(
            "Baixar resultados por frame (CSV)",
            data=res["df"].to_csv(index=False).encode("utf-8"),
            file_name="per_frame_results.csv",
            mime="text/csv"
        )
        st.download_button(
            "Baixar resumo (CSV)",
            data=res["summary"].to_csv(index=False).encode("utf-8"),
            file_name="summary_metrics.csv",
            mime="text/csv"
        )

        st.caption(f"Arquivos salvos em: {res['run_dir']}")
        st.caption(f"Tempo total: {res['time_sec']:.1f}s")

        st.markdown("**Observações técnicas**")
        st.markdown("""
- As **classes** são lidas de `classes.txt` ou inferidas; se não houver arquivo, usa o conjunto padrão:  
  `['DeepFakeDetection','FaceShifter','FaceSwap','NeuraTextures','deepfakes','face2face','original']` (cortado/expandido conforme `num_classes`).  
- A camada padrão para Grad-CAM é `layer4` (ResNet-50). Ajuste se o seu modelo tiver topologia diferente.  
- A segmentação de rosto usa **HaarCascade**; se não detectar rosto, usa o frame completo.  
- Métricas de XAI: *Saliency Entropy*, *Salience-assessed Reaction to Noise*, *Salience Resilience to Geometrical Transformations* e *Accuracy over SLIC-based Segmentation*.  
        """)

# ===================== CLI =====================
def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Deepfake Forensics - CLI")
    parser.add_argument("--model", required=True, help="Caminho do .pth")
    parser.add_argument("--video", required=True, help="Caminho do vídeo")
    parser.add_argument("--frames", type=int, default=32, help="Número de frames")
    parser.add_argument("--segment_face", action="store_true", help="Ativar segmentação de rosto")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Pasta de saída")
    parser.add_argument("--target_layer", default="layer4", help="Camada alvo para Grad-CAM (ResNet)")
    parser.add_argument("--slic_segments", type=int, default=50, help="SLIC n_segments")
    parser.add_argument("--slic_compactness", type=int, default=20, help="SLIC compactness")
    parser.add_argument("--no_save_images", action="store_true", help="Não salvar PNGs dos painéis")

    args = parser.parse_args()

    def on_progress(msg):
        print(f"[INFO] {msg}", flush=True)

    try:
        res = process_video(
            model_path=args.model,
            video_path=args.video,
            num_frames=int(args.frames),
            segment_face=args.segment_face,
            outdir=args.outdir,
            target_layer=args.target_layer,
            save_images=(not args.no_save_images),
            slic_segments=int(args.slic_segments),
            slic_compactness=int(args.slic_compactness),
            ui_callbacks={"on_progress": on_progress}
        )
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        return

    print(json.dumps({
        "status": "ok",
        "run_dir": res["run_dir"],
        "time_sec": round(res["time_sec"], 3),
        "num_classes": res["num_classes"]
    }, ensure_ascii=False))

# ===================== ENTRYPOINT =====================
if __name__ == "__main__":
    if _is_streamlit_context():
        run_streamlit()
    else:
        run_cli()
