# app_inferencia.py
# Deepfake Forensics – Classificação + XAI (Original | SLIC | Grad-CAM | SmoothGrad)
# Reescrito com robustez de I/O, logs, meta.json e salvamento parcial.

import os, cv2, time, glob, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic, mark_boundaries
from scipy.stats import entropy

# ===================== Config =====================
DEFAULT_MODELS_DIR = r"C:\MODELS"
DEFAULT_VIDEOS_DIR = r"C:\VIDEOS"
DEFAULT_OUTPUT_DIR = r"C:\MODELS_OUTPUTS"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Streamlit detector =====================
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

# ===================== Captum (opcional) =====================
_HAS_CAPTUM = False
try:
    from captum.attr import Saliency, NoiseTunnel
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False

# ===================== Utils =====================
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def _safe_norm01(arr):
    arr = np.asarray(arr)
    m = np.nanmax(arr)
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    out = arr / m
    out[~np.isfinite(out)] = 0
    return out.astype(np.float32)

def pil_from_bgr(img_bgr):  # BGR -> PIL RGB
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def tensor_from_pil(img_pil, device=DEVICE):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return preprocess(img_pil).unsqueeze(0).to(device)

def overlay_heatmap_on_rgb(rgb224, heatmap):
    hm = cv2.resize(heatmap, (rgb224.shape[1], rgb224.shape[0]))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return (0.6 * rgb224 + 0.4 * hm_color).astype(np.uint8)

# ===================== Modelo =====================
def infer_num_classes_from_state_dict(sd: dict) -> int:
    for k in ["fc.weight", "classifier.weight", "head.weight"]:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 2:
            return sd[k].shape[0]
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
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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
    model.to(DEVICE).eval()
    return model, num_classes

def maybe_load_class_names(model_path, num_classes):
    base = os.path.splitext(model_path)[0]
    for p in [base + ".classes.txt", os.path.join(os.path.dirname(model_path), "classes.txt")]:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) >= 2:
                return lines[:num_classes]
    default = ["DeepFakeDetection","FaceShifter","FaceSwap","NeuraTextures","deepfakes","face2face","original"]
    if num_classes <= len(default): return default[:num_classes]
    return default + [f"class_{i}" for i in range(len(default), num_classes)]

# ===================== XAI: Grad-CAM =====================
def make_gradcam_pytorch(model, img_tensor, target_layer='layer4'):
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_module = None
    for name, module in model.named_modules():
        if name == target_layer:
            target_module = module
            break
    if target_module is None:
        raise RuntimeError(f"Camada alvo '{target_layer}' não encontrada no modelo.")

    target_module.register_forward_hook(forward_hook)
    target_module.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = torch.argmax(output, dim=1)
    loss = output[0, class_idx]
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = gradients[0]
    act = activations[0]
    pooled_grad = torch.mean(grad, dim=[0, 2, 3])
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_grad[i]
    heatmap = act.mean(dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0.0)
    heatmap = _safe_norm01(heatmap)
    return heatmap, output.detach()

# ===================== XAI: SmoothGrad =====================
def make_smoothgrad(model, img_tensor, nt_samples=20, stdevs=0.2):
    """
    SmoothGrad com target = classe prevista (argmax).
    Captum se disponível; senão, fallback manual.
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        target_idx = int(torch.argmax(logits, dim=1).item())

    if _HAS_CAPTUM:
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attr = nt.attribute(
            img_tensor,
            target=target_idx,
            nt_type='smoothgrad',
            nt_samples=nt_samples,
            stdevs=stdevs
        )
        attr = attr.squeeze().detach().cpu().numpy()  # [C,H,W]
        attr = np.abs(attr).mean(axis=0)
        return _safe_norm01(attr)

    grads = []
    for _ in range(nt_samples):
        noisy = (img_tensor + torch.normal(0, stdevs, img_tensor.shape).to(img_tensor.device)
                 ).clone().detach().requires_grad_(True)
        out = model(noisy)
        loss = out[0, target_idx]
        model.zero_grad(set_to_none=True)
        if noisy.grad is not None:
            noisy.grad.zero_()
        loss.backward()
        grads.append(noisy.grad.detach().clone())
    grad_avg = torch.mean(torch.stack(grads), dim=0)  # [1,C,H,W]
    smap = grad_avg.abs().mean(dim=1).squeeze().cpu().numpy()
    return _safe_norm01(smap)

# ===================== Métricas =====================
def saliency_entropy(smap):
    smap = smap / (np.sum(smap) + 1e-12)
    return float(entropy(smap.flatten() + 1e-12))

def reaction_to_noise(model, img_tensor, original_salmap, target_layer='layer4'):
    noisy = img_tensor + torch.normal(0, 0.05, img_tensor.shape).to(DEVICE)
    noisy_salmap, _ = make_gradcam_pytorch(model, noisy, target_layer=target_layer)
    return float(np.mean(np.abs(original_salmap - noisy_salmap)))

def geometrical_robustness(model, raw_img_rgb_224, original_salmap, target_layer='layer4'):
    flipped = np.flip(raw_img_rgb_224, axis=1).copy()
    flipped_tensor = tensor_from_pil(Image.fromarray(flipped))
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
        t_ori = tensor_from_pil(Image.fromarray(img_uint8))
        t_deg = tensor_from_pil(Image.fromarray(degraded))
        logits_ori = model(t_ori)[0].detach().cpu().numpy()
        logits_deg = model(t_deg)[0].detach().cpu().numpy()
    aos = float(np.abs(logits_ori - logits_deg).mean())
    return aos, segs

def slic_with_boundaries(rgb224, segments=50, compactness=20):
    img_uint8 = rgb224.astype(np.uint8)
    segs = slic(img_uint8, n_segments=segments, compactness=compactness, enforce_connectivity=True)
    vis = mark_boundaries(img_uint8.astype(np.float32)/255.0, segs, color=(1,1,0))
    return segs, (vis*255).astype(np.uint8)

# ===================== Vídeo / Frames =====================
def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_largest_face_bgr(frame_bgr, scale_factor=1.1, min_neighbors=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_detector().detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if len(faces) == 0: return frame_bgr
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    pad = int(0.10 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w + pad); y1 = min(frame_bgr.shape[0], y + h + pad)
    return frame_bgr[y0:y1, x0:x1, :]

def sample_frames_uniform(video_path, num_samples):
    # 1) OpenCV (prefer FFMPEG)
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    except Exception:
        cap = cv2.VideoCapture(video_path)

    if not cap or not cap.isOpened():
        # 2) Fallback imageio
        try:
            import imageio.v3 as iio
            frames = []
            # tentar obter n_frames
            nframes = None
            try:
                meta = iio.immeta(video_path)
                nframes = meta.get("n_frames")
            except Exception:
                pass
            idxs = None
            if nframes and nframes > 0:
                idxs = set(np.linspace(0, nframes - 1, num=num_samples, dtype=int).tolist())
            k = 0
            for fr in iio.imiter(video_path):
                if fr.ndim == 2:
                    fr = np.stack([fr, fr, fr], axis=-1)
                fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                if idxs is None or k in idxs:
                    frames.append(fr_bgr)
                    if len(frames) >= num_samples: break
                k += 1
            return frames
        except Exception as e:
            print(f"[ERROR] Falha ao abrir vídeo: {e}", flush=True)
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
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames

# ===================== Pipeline =====================
def process_video(
    model_path, video_path, num_frames, segment_face, outdir,
    target_layer="layer4", save_images=True,
    slic_segments=50, slic_compactness=20,
    enable_smoothgrad=True, smooth_nt_samples=20, smooth_stdevs=0.2,
    max_preview=6, class_names=None, ui_callbacks=None
):
    t0 = time.time()
    ensure_dir(outdir)
    run_name = f"{os.path.splitext(os.path.basename(video_path))[0]}__{os.path.splitext(os.path.basename(model_path))[0]}__{int(time.time())}"
    run_dir = os.path.join(outdir, run_name)
    imgs_dir = os.path.join(run_dir, "frames"); ensure_dir(imgs_dir)

    # meta.json inicial
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "video_path": video_path,
            "start_time": int(time.time()),
            "device": DEVICE
        }, f, ensure_ascii=False, indent=2)

    if ui_callbacks and ui_callbacks.get("on_progress"):
        ui_callbacks["on_progress"]("Carregando modelo...")
        ui_callbacks["on_progress"](f"Run dir: {run_dir}")
    print(f"[INFO] Run dir: {run_dir}", flush=True)

    model, num_classes = load_model_auto(model_path)
    if class_names is None: class_names = maybe_load_class_names(model_path, num_classes)

    if ui_callbacks and ui_callbacks.get("on_progress"):
        ui_callbacks["on_progress"](f"Modelo carregado. Classes: {num_classes}")
        ui_callbacks["on_progress"](f"Classes detectadas: {class_names}")
        status = "desativado"
        if enable_smoothgrad: status = "Captum" if _HAS_CAPTUM else "fallback"
        ui_callbacks["on_progress"](f"SmoothGrad: {status} (samples={smooth_nt_samples}, std={smooth_stdevs})")

    frames_bgr = sample_frames_uniform(video_path, num_frames)
    if not frames_bgr:
        raise RuntimeError("Não foi possível extrair frames do vídeo (codec/caminho?).")

    results = []
    partial_csv = os.path.join(run_dir, "per_frame_results_partial.csv")

    try:
        for idx, frame_bgr in enumerate(frames_bgr, start=1):
            if segment_face:
                frame_bgr = crop_largest_face_bgr(frame_bgr)

            pil_img = pil_from_bgr(frame_bgr).resize((224, 224))
            rgb224 = np.array(pil_img)
            img_tensor = tensor_from_pil(pil_img)

            # Grad-CAM + logits
            salmap, logits = make_gradcam_pytorch(model, img_tensor, target_layer=target_layer)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            top3_idx = probs.argsort()[-3:][::-1]
            top3 = [(int(i), float(probs[i]), class_names[int(i)] if int(i) < len(class_names) else f"class_{int(i)}")
                    for i in top3_idx]

            # SLIC + métricas
            _, slic_vis = slic_with_boundaries(rgb224, segments=slic_segments, compactness=slic_compactness)
            ent = saliency_entropy(salmap)
            r_noise = reaction_to_noise(model, img_tensor, salmap, target_layer=target_layer)
            r_geo = geometrical_robustness(model, rgb224, salmap, target_layer=target_layer)
            aos, _ = accuracy_over_segmentation(model, rgb224, salmap, segments=slic_segments, compactness=slic_compactness)

            gradcam_overlay = overlay_heatmap_on_rgb(rgb224, salmap)

            # SmoothGrad
            smooth_overlay, smooth_engine = None, "disabled"
            smooth_path = ""
            if enable_smoothgrad:
                smooth_engine = "captum" if _HAS_CAPTUM else "fallback"
                smooth_map = make_smoothgrad(model, img_tensor, nt_samples=smooth_nt_samples, stdevs=smooth_stdevs)
                smooth_overlay = overlay_heatmap_on_rgb(rgb224, smooth_map)

            # Salvar imagens
            orig_path = os.path.join(imgs_dir, f"frame_{idx:04d}_orig.png")
            slic_path = os.path.join(imgs_dir, f"frame_{idx:04d}_slic.png")
            cam_path  = os.path.join(imgs_dir, f"frame_{idx:04d}_gradcam.png")
            if enable_smoothgrad:
                smooth_path = os.path.join(imgs_dir, f"frame_{idx:04d}_smooth.png")

            if save_images:
                Image.fromarray(rgb224).save(orig_path)
                Image.fromarray(slic_vis).save(slic_path)
                Image.fromarray(gradcam_overlay).save(cam_path)
                if enable_smoothgrad and smooth_overlay is not None:
                    Image.fromarray(smooth_overlay).save(smooth_path)

            # Registro
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
                "gradcam_path": cam_path if save_images else "",
                "smoothgrad_path": (smooth_path if (save_images and enable_smoothgrad and smooth_overlay is not None) else ""),
                "smoothgrad_enabled": bool(enable_smoothgrad),
                "smoothgrad_engine": smooth_engine
            })

        # Salva finais
        df = pd.DataFrame(results)
        summary = df[["saliency_entropy","salience_reaction_to_noise",
                      "salience_resilience_geometric","accuracy_over_slic"]].mean().to_frame("mean").T
        df.to_csv(os.path.join(run_dir, "per_frame_results.csv"), index=False, encoding="utf-8")
        summary.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False, encoding="utf-8")

        return {"run_dir": run_dir, "df": df, "summary": summary, "time_sec": time.time()-t0, "num_classes": len(class_names)}

    except Exception as e:
        # Salva parcial para depuração
        if results:
            pd.DataFrame(results).to_csv(partial_csv, index=False, encoding="utf-8")
        raise e

    finally:
        # fecha meta.json
        try:
            with open(meta_path, "r+", encoding="utf-8") as f:
                meta = json.load(f)
            meta["end_time"] = int(time.time())
            meta["frames_processed"] = len(results)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ===================== UI (Streamlit) =====================
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Deepfake Forensics App", layout="wide")
    st.title("🔍 Deepfake Forensics – Original | SLIC | Grad-CAM | SmoothGrad")
    st.caption(f"Dispositivo: **{DEVICE}**")

    with st.sidebar:
        st.header("Configurações")
        models_dir = st.text_input("Pasta de modelos (.pth)", DEFAULT_MODELS_DIR)
        videos_dir = st.text_input("Pasta de vídeos", DEFAULT_VIDEOS_DIR)
        output_dir = st.text_input("Pasta de saída (resultados)", DEFAULT_OUTPUT_DIR)

        model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
        if not model_paths: st.warning("Nenhum .pth encontrado (ajuste o caminho).")
        model_path = st.selectbox("Escolha o modelo (.pth)", model_paths, index=0 if model_paths else None)

        video_exts = ("*.mp4","*.avi","*.mov","*.mkv","*.wmv")
        video_files = []
        if os.path.isdir(videos_dir):
            for ext in video_exts: video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        video_files = sorted(video_files)
        if not video_files: st.warning("Nenhum vídeo encontrado (ajuste o caminho).")
        video_path = st.selectbox("Escolha o vídeo", video_files, index=0 if video_files else None)

        num_frames = st.number_input("Número de frames (amostragem uniforme)", min_value=1, max_value=2000, value=32, step=1)
        segment_face = st.checkbox("Segmentar rosto (HaarCascade)", value=True)
        target_layer = st.text_input("Camada Grad-CAM (ResNet: layer4 | EfficientNet: features.6)", value="layer4")
        slic_segments = st.slider("SLIC - n_segments", 20, 300, 50, 5)
        slic_compactness = st.slider("SLIC - compactness", 1, 40, 20, 1)

        st.divider(); st.subheader("SmoothGrad")
        enable_smoothgrad = st.checkbox("Ativar SmoothGrad", value=True)
        smooth_nt_samples = st.slider("nt_samples", 5, 64, 20, 1)
        smooth_stdevs = st.slider("stdevs (ruído)", 0.05, 0.50, 0.20, 0.05)

        save_images = st.checkbox("Salvar PNGs dos painéis no disco", value=True)
        run_btn = st.button("▶️ Executar")

    placeholder_progress = st.empty()
    cols = st.columns(4); shown = {"n": 0}

    def on_progress(msg): placeholder_progress.info(msg)

    def on_quad(img_orig, img_slic, img_cam, img_smooth, caption):
        cols[0].image(img_orig,  caption="Original (224x224)", use_column_width=True)
        cols[1].image(img_slic,  caption="SLIC (contornos)",   use_column_width=True)
        cols[2].image(img_cam,   caption="Grad-CAM",           use_column_width=True)
        cols[3].image(img_smooth if img_smooth is not None else img_cam,
                      caption=caption or "SmoothGrad", use_column_width=True)
        shown["n"] += 1

    if run_btn and model_path and video_path:
        try:
            res = process_video(
                model_path=model_path, video_path=video_path, num_frames=int(num_frames),
                segment_face=segment_face, outdir=output_dir, target_layer=target_layer,
                save_images=save_images, slic_segments=int(slic_segments), slic_compactness=int(slic_compactness),
                enable_smoothgrad=enable_smoothgrad, smooth_nt_samples=int(smooth_nt_samples),
                smooth_stdevs=float(smooth_stdevs),
                ui_callbacks={"on_progress": on_progress, "on_quad": on_quad}
            )
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            return

        st.success("Processamento finalizado.")
        st.subheader("📊 Resumo das métricas (média por vídeo)")
        st.dataframe(res["summary"], use_container_width=True)

        st.download_button("Baixar resultados por frame (CSV)", data=res["df"].to_csv(index=False).encode("utf-8"),
                           file_name="per_frame_results.csv", mime="text/csv")
        st.download_button("Baixar resumo (CSV)", data=res["summary"].to_csv(index=False).encode("utf-8"),
                           file_name="summary_metrics.csv", mime="text/csv")

        st.caption(f"Arquivos salvos em: {res['run_dir']}")
        st.caption(f"Tempo total: {res['time_sec']:.1f}s")

# ===================== CLI =====================
def run_cli():
    import argparse
    p = argparse.ArgumentParser(description="Deepfake Forensics - CLI")
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--segment_face", action="store_true")
    p.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target_layer", default="layer4", help="Ex.: layer4 (ResNet) | features.6 (EfficientNet)")
    p.add_argument("--slic_segments", type=int, default=50)
    p.add_argument("--slic_compactness", type=int, default=20)
    p.add_argument("--enable_smoothgrad", action="store_true")
    p.add_argument("--smooth_nt_samples", type=int, default=20)
    p.add_argument("--smooth_stdevs", type=float, default=0.20)
    p.add_argument("--no_save_images", action="store_true")
    args = p.parse_args()

    def on_progress(msg): print(f"[INFO] {msg}", flush=True)

    try:
        res = process_video(
            model_path=args.model, video_path=args.video, num_frames=int(args.frames),
            segment_face=args.segment_face, outdir=args.outdir, target_layer=args.target_layer,
            save_images=(not args.no_save_images), slic_segments=int(args.slic_segments),
            slic_compactness=int(args.slic_compactness),
            enable_smoothgrad=bool(args.enable_smoothgrad), smooth_nt_samples=int(args.smooth_nt_samples),
            smooth_stdevs=float(args.smooth_stdevs),
            ui_callbacks={"on_progress": on_progress}
        )
    except Exception as e:
        print(json.dumps({"status":"error","message":str(e)}, ensure_ascii=False))
        sys.exit(1)

    print(json.dumps({"status":"ok","run_dir":res["run_dir"],"time_sec":round(res["time_sec"],3),
                      "num_classes":res["num_classes"]}, ensure_ascii=False))

# ===================== Entry =====================
if __name__ == "__main__":
    if _is_streamlit_context():
        run_streamlit()
    else:
        run_cli()
