import os
import json
import time
import threading
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_PATH", "/data/training"))
_SAMPLES_FILE = TRAINING_DIR / "samples.jsonl"
_BIAS_FILE = TRAINING_DIR / "bias.json"
_AGE_GROUP_BIAS_FILE = TRAINING_DIR / "age_group_bias.json"
_MIN_SAMPLES_TO_FINETUNE = 5
_AGE_GROUPS = [(0, 20, "0-20"), (21, 40, "21-40"), (41, 60, "41-60"), (61, 120, "61+")]


def _get_age_group(age: int) -> str:
    for lo, hi, label in _AGE_GROUPS:
        if lo <= age <= hi:
            return label
    return "61+"


def update_age_group_bias(real_age: int, raw_predicted_age: int) -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(_AGE_GROUP_BIAS_FILE.read_text()) if _AGE_GROUP_BIAS_FILE.exists() else {}
    except Exception:
        data = {}
    group = _get_age_group(real_age)
    entry = data.get(group, {"bias": 0.0, "n": 0})
    n, old_bias = entry["n"], entry["bias"]
    new_bias = round((old_bias * n + (real_age - raw_predicted_age)) / (n + 1), 2)
    data[group] = {"bias": new_bias, "n": n + 1}
    _AGE_GROUP_BIAS_FILE.write_text(json.dumps(data, indent=2))
    print(f"[trainer] Age group '{group}' bias: {new_bias:+.2f} yrs ({n + 1} samples)")


def load_age_group_bias() -> dict:
    """Return {group_label: bias_float} for all age groups."""
    if not _AGE_GROUP_BIAS_FILE.exists():
        return {}
    try:
        data = json.loads(_AGE_GROUP_BIAS_FILE.read_text())
        return {k: float(v["bias"]) for k, v in data.items()}
    except Exception:
        return {}

_finetune_lock = threading.Lock()
_finetune_running = False


def save_sample(session_id: str, real_age: int, predicted_age, frame_paths: list) -> dict:
    """Persist one feedback sample and recompute bias correction."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SAMPLES_FILE, "a") as f:
        f.write(json.dumps({
            "session_id": session_id,
            "real_age": real_age,
            "predicted_age": predicted_age,
            "frame_paths": [str(p) for p in frame_paths],
            "timestamp": time.time(),
        }) + "\n")
    samples = _load_samples()
    bias = _recompute_bias(samples)
    return {
        "n_samples": len(samples),
        "bias": bias,
        "can_finetune": len(samples) >= _MIN_SAMPLES_TO_FINETUNE,
    }


def load_bias() -> float:
    if not _BIAS_FILE.exists():
        return 0.0
    try:
        return float(json.loads(_BIAS_FILE.read_text()).get("bias", 0.0))
    except Exception:
        return 0.0


def start_finetune_async(pipeline) -> bool:
    """Launch fine-tuning in a background thread. Returns True if started."""
    global _finetune_running
    with _finetune_lock:
        if _finetune_running:
            return False
        samples = _load_samples()
        valid = [s for s in samples if any(os.path.exists(fp) for fp in s.get("frame_paths", []))]
        if len(valid) < _MIN_SAMPLES_TO_FINETUNE:
            return False
        _finetune_running = True
    threading.Thread(target=_run_finetune, args=(pipeline, samples), daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_samples() -> list:
    if not _SAMPLES_FILE.exists():
        return []
    out = []
    for line in _SAMPLES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _recompute_bias(samples: list) -> float:
    valid = [s for s in samples if s.get("predicted_age") is not None]
    if not valid:
        return 0.0
    bias = round(sum(s["real_age"] - s["predicted_age"] for s in valid) / len(valid), 2)
    _BIAS_FILE.write_text(json.dumps({"bias": bias, "n_samples": len(valid)}))
    print(f"[trainer] Bias correction updated: {bias:+.1f} yrs ({len(valid)} samples)")
    return bias


def _run_finetune(pipeline, samples: list) -> None:
    global _finetune_running
    try:
        import cv2
        import torch
        import torch.nn.functional as F

        age_det = pipeline.age_detector
        if age_det is None:
            print("[trainer] No age detector available, skipping fine-tune")
            return

        # Build (image, real_age) pairs — one frame per session
        pairs = []
        for s in samples:
            real_age = float(s["real_age"])
            for fp in s.get("frame_paths", []):
                if fp and os.path.exists(fp):
                    img = cv2.imread(fp)
                    if img is not None and img.size > 0:
                        pairs.append((img, real_age))
                        break

        if not pairs:
            print("[trainer] No valid face images found, skipping fine-tune")
            return

        print(f"[trainer] Fine-tuning on {len(pairs)} samples…")
        model = age_det._model
        processor = age_det._processor

        # Hold model lock for the full training run to avoid racing with inference
        with age_det._model_lock:
            for p in model.parameters():
                p.requires_grad = False
            # Unfreeze the last 10 parameter tensors (regression head)
            for _, p in list(model.named_parameters())[-10:]:
                p.requires_grad = True

            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                print("[trainer] Could not identify trainable parameters, skipping")
                return

            opt = torch.optim.Adam(trainable, lr=5e-5)
            model.train()
            try:
                for epoch in range(5):
                    total = 0.0
                    for img, real_age in pairs:
                        inputs = processor([img, None])
                        pv = inputs.pixel_values
                        out = model(faces_input=pv[0:1], body_input=pv[1:2], return_dict=True)
                        target = torch.tensor([[real_age]], dtype=torch.float32)
                        loss = F.mse_loss(out.age_output.float(), target)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total += loss.item()
                    mae = (total / len(pairs)) ** 0.5
                    print(f"[trainer] Epoch {epoch + 1}/5 — MAE ≈ {mae:.1f} yrs")
            finally:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

        print("[trainer] Fine-tuning complete.")

        # Delete face images — they were only needed for training
        deleted = 0
        for s in samples:
            for fp in s.get("frame_paths", []):
                if fp and os.path.exists(fp):
                    try:
                        os.remove(fp)
                        deleted += 1
                    except OSError:
                        pass
        if deleted:
            print(f"[trainer] Deleted {deleted} face image(s) after fine-tuning.")
    except Exception as e:
        print(f"[trainer] Fine-tune error: {e}")
    finally:
        _finetune_running = False
