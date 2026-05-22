import os
import sys
import cv2
import queue
import threading
import time

# YOLOv8 is optional — face detection still works without it
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Glasses classifier + YOLOS watch detector (transformers + torchvision) are optional
try:
    import torch
    import torchvision.transforms as T
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        YolosImageProcessor,
        YolosForObjectDetection,
    )
    from PIL import Image as PILImage
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Face detection constants (unchanged)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
BOX_COLOR   = (0, 215, 255)   # gold/yellow box
BADGE_COLOR = (0, 215, 255)   # badge background
TEXT_COLOR  = (0, 0, 0)       # black text on badge

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROTOTXT   = os.path.join(BASE_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ---------------------------------------------------------------------------
# Accessory detection constants (new)
# ---------------------------------------------------------------------------
ACCESSORY_CONF  = 0.45                                  # min confidence for face accessories
ACCESSORY_MODEL = os.path.join(BASE_DIR, "accessories.pt")  # YOLOv8 weights file
ROI_MIN_PX      = 60    # skip accessory detection for faces narrower than this
ROI_PAD         = 20    # extra pixels around face ROI improves hat/glasses detection
WATCH_CONF       = 0.28  # min confidence for watch detection
WATCH_IMGSZ      = 1280  # larger input → model sees more detail → better small-object detection
WATCH_MODEL_NAME = "yolov8n-oiv7.pt"  # Open Images V7 nano — has real-world Watch class

# Age detection — iitolstykh/mivolo_v2 (MiVOLO, predicts exact integer age)
AGE_MODEL_ID = "iitolstykh/mivolo_v2"

# Fashion detection — valentinafevu/yolos-fashionpedia (46 classes, full frame)
FASHION_CONF     = 0.45
FASHION_MODEL_ID = "valentinafevu/yolos-fashionpedia"
# Fashionpedia label → display name. Glasses and watch are excluded (dedicated detectors).
FASHION_TARGETS = {
    "hat":                                       "Hat",
    "headband, head covering, hair accessory":   "Headband",
    "tie":                                       "Tie",
    "scarf":                                     "Scarf",
    "glove":                                     "Glove",
    "belt":                                      "Belt",
    "bag, wallet":                               "Bag",
    "umbrella":                                  "Umbrella",
}

# Class names from keremberke/yolov8m-protective-equipment-detection (accessories.pt).
# The model's full class list: glove, goggles, helmet, mask, no_glove, no_goggles,
# no_helmet, no_mask, no_shoes, shoes  — "no_*" variants are intentionally excluded.
# Hat/Cap is not in this model; aliases are kept so a future model swap just works.
TARGET_ACCESSORIES = {
    "Glasses":   ["goggles"],
    "Face Mask": ["mask"],
    "Hat/Cap":   ["hat", "cap", "beanie", "baseball cap", "beret"],
    "Helmet":    ["helmet"],
}

# Per-accessory badge colour (BGR)
ACCESSORY_COLORS = {
    "Glasses":   (50, 200, 50),
    "Face Mask": (50, 150, 255),
    "Hat/Cap":   (200, 50, 200),
    "Helmet":    (50, 50, 220),
    "Watch":     (0, 230, 200),
    # Fashion detector colours
    "Hat":       (0, 120, 255),
    "Headband":  (180, 60, 255),
    "Tie":       (100, 180, 80),
    "Scarf":     (0, 200, 180),
    "Glove":     (30, 100, 160),
    "Belt":      (0, 165, 255),
    "Bag":       (130, 60, 200),
    "Umbrella":  (200, 200, 0),
}


# ---------------------------------------------------------------------------
# Accessory detector (YOLOv8 wrapper)
# ---------------------------------------------------------------------------

class AccessoryDetector:
    """
    Runs YOLOv8 + Haar cascade on face-ROI crops in a background thread.
    Results are (category, conf, frame_x1, frame_y1, frame_x2, frame_y2) tuples
    so the caller can draw bounding boxes at the exact detection location.
    """

    def __init__(self, model_path: str):
        self._model = YOLO(model_path)
        self._lookup: dict[str, str] = {}
        for category, aliases in TARGET_ACCESSORIES.items():
            for alias in aliases:
                self._lookup[alias.lower()] = category

        # Binary glasses classifier — youngp5/eyeglasses_detection
        # labels: {0: 'glasses', 1: 'no glasses'}
        # AutoImageProcessor is broken for this model; use manual ImageNet preprocessing.
        if TRANSFORMERS_AVAILABLE:
            self._glasses_model = AutoModelForImageClassification.from_pretrained(
                "youngp5/eyeglasses_detection"
            )
            self._glasses_model.eval()
            self._glasses_tf = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self._glasses_model = None
            print("transformers/torchvision not installed — glasses detection disabled.")
            print("  Install:  pip install transformers torchvision Pillow")

        self._queue   = queue.Queue(maxsize=1)
        self._results : list = []
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Called from the main (display) thread — never blocks
    # ------------------------------------------------------------------

    def submit(self, roi_bgr, roi_x1: int, roi_y1: int) -> None:
        """Queue a face ROI + its top-left frame offset. Drops if worker is busy."""
        if roi_bgr is None or roi_bgr.size == 0:
            return
        try:
            self._queue.put_nowait((roi_bgr.copy(), roi_x1, roi_y1))
        except queue.Full:
            pass

    def get_results(self) -> list:
        """Return last known detections as [(category, conf, x1, y1, x2, y2), ...]."""
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    # ------------------------------------------------------------------
    # Background worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            roi, rx1, ry1 = item
            results = self._infer(roi, rx1, ry1)
            with self._lock:
                self._results = results
            time.sleep(0.04)  # cap at ~25 fps — yields CPU to the display loop

    def _infer(self, roi_bgr, rx1: int, ry1: int) -> list:
        """Run YOLO + Haar inference and return frame-space bounding boxes."""
        detections = self._model(roi_bgr, conf=ACCESSORY_CONF, verbose=False)
        # best: category → (conf, fx1, fy1, fx2, fy2)
        best: dict[str, tuple] = {}

        for result in detections:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id].lower()
                conf     = float(box.conf[0])
                category = self._lookup.get(cls_name)
                if category:
                    bx1, by1, bx2, by2 = (int(v) for v in box.xyxy[0])
                    frame_coords = (rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2)
                    if category not in best or conf > best[category][0]:
                        best[category] = (conf, *frame_coords)

        # Glasses: binary classifier — label index 0 = 'glasses', 1 = 'no glasses'
        if "Glasses" not in best and self._glasses_model is not None:
            roi_rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            tensor   = self._glasses_tf(PILImage.fromarray(roi_rgb)).unsqueeze(0)
            with torch.no_grad():
                conf = torch.softmax(self._glasses_model(tensor).logits, dim=-1)[0][0].item()
            if conf >= ACCESSORY_CONF:
                # Classifier gives no box — estimate eye-band from face proportions
                rh, rw   = roi_bgr.shape[:2]
                ex1, ey1 = int(rw * 0.10), int(rh * 0.25)
                ex2, ey2 = int(rw * 0.90), int(rh * 0.55)
                best["Glasses"] = (conf, rx1+ex1, ry1+ey1, rx1+ex2, ry1+ey2)

        return [(cat, *vals) for cat, vals in best.items()]


# ---------------------------------------------------------------------------
# Watch detector (YOLOS-Fashionpedia, runs on the full frame)
# ---------------------------------------------------------------------------

class WatchDetector:
    """
    Uses YOLOv8n trained on Open Images V7 (600 real-world classes, includes 'Watch')
    to detect watches anywhere in the full frame. Runs in a background thread so the
    display loop never blocks. The model auto-downloads via ultralytics on first run.
    """

    def __init__(self):
        print(f"Loading watch/hat model ({WATCH_MODEL_NAME}) …")
        self._model = YOLO(WATCH_MODEL_NAME)  # downloads to ultralytics cache if absent

        # Watch class IDs
        self._watch_ids = [
            idx for idx, name in self._model.names.items()
            if "watch" in name.lower()
        ]
        # Hat class IDs — covers Hat, Cowboy hat, Fedora, Sun hat, Swim cap
        HAT_KEYWORDS = ("hat", "cap", "fedora", "beret", "beanie")
        self._hat_ids = [
            idx for idx, name in self._model.names.items()
            if any(k in name.lower() for k in HAT_KEYWORDS)
            and "helmet" not in name.lower()  # helmets handled by PPE model
        ]
        self._all_ids = self._watch_ids + self._hat_ids
        print(f"Watch classes : {[self._model.names[i] for i in self._watch_ids]}")
        print(f"Hat classes   : {[self._model.names[i] for i in self._hat_ids]}")

        self._queue   = queue.Queue(maxsize=1)
        self._results : list = []
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr) -> None:
        try:
            self._queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass

    def get_results(self) -> list:
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            with self._lock:
                self._results = self._infer(frame)
            time.sleep(0.15)  # cap at ~6 fps — imgsz=1280 is expensive; yield CPU

    def _infer(self, frame_bgr) -> list:
        results = self._model(
            frame_bgr, conf=WATCH_CONF,
            classes=self._all_ids,  # Watch + Hat family — skip all other 590+ classes
            imgsz=WATCH_IMGSZ,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                label = "Watch" if cls_id in self._watch_ids else "Hat"
                detections.append((label, conf, x1, y1, x2, y2))
        return detections


# ---------------------------------------------------------------------------
# Fashion detector (YOLOS-Fashionpedia, runs on the full frame)
# ---------------------------------------------------------------------------

class FashionDetector:
    """
    Uses valentinafevu/yolos-fashionpedia (46 fashion classes) to detect
    wearable accessories on the full frame. Glasses and watch are excluded
    — those have dedicated detectors. Runs in a background thread.
    """

    def __init__(self):
        print(f"Loading fashion model ({FASHION_MODEL_ID}) …")
        self._processor = YolosImageProcessor.from_pretrained(FASHION_MODEL_ID)
        self._model     = YolosForObjectDetection.from_pretrained(FASHION_MODEL_ID)
        self._model.eval()

        # Map model label index → display name for the classes we care about
        self._label_map: dict[int, str] = {}
        for idx, label in self._model.config.id2label.items():
            display = FASHION_TARGETS.get(label.lower())
            if display:
                self._label_map[idx] = display

        print(f"Fashion classes active: {sorted(set(self._label_map.values()))}")

        self._queue   = queue.Queue(maxsize=1)
        self._results : list = []
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr) -> None:
        try:
            self._queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass

    def get_results(self) -> list:
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            with self._lock:
                self._results = self._infer(frame)
            time.sleep(0.20)  # ~5 fps max — transformer on full frame is heavy; yield CPU

    def _infer(self, frame_bgr) -> list:
        h, w    = frame_bgr.shape[:2]
        pil_img = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs  = self._processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs, threshold=FASHION_CONF,
            target_sizes=torch.tensor([[h, w]])
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            display = self._label_map.get(label.item())
            if display:
                x1, y1, x2, y2 = (int(v) for v in box.tolist())
                detections.append((display, float(score), x1, y1, x2, y2))
        return detections


# ---------------------------------------------------------------------------
# Age detector (MiVOLO, runs on face ROI in background thread)
# ---------------------------------------------------------------------------

class AgeDetector:
    """
    Predicts exact age from a face ROI using MiVOLO (iitolstykh/mivolo_v2).
    Returns an integer age string, e.g. "27".
    Runs in a background thread; never blocks the display loop.
    """

    def __init__(self):
        print(f"Loading age model ({AGE_MODEL_ID}) …")
        self._processor = AutoImageProcessor.from_pretrained(AGE_MODEL_ID, trust_remote_code=True)
        self._model     = AutoModelForImageClassification.from_pretrained(
            AGE_MODEL_ID, trust_remote_code=True, dtype=torch.float32
        )
        self._model.eval()
        self._queue  = queue.Queue(maxsize=1)
        self._result = None
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, face_roi_bgr) -> None:
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return
        try:
            self._queue.put_nowait(face_roi_bgr.copy())
        except queue.Full:
            pass

    def get_result(self):
        """Return age string or None if no result yet."""
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            roi = self._queue.get()
            if roi is None:
                break
            inputs = self._processor([roi, None])  # [face_crop, no_body_crop]
            pv = inputs.pixel_values               # [2, 3, 384, 384]
            with torch.no_grad():
                out = self._model(faces_input=pv[0:1], body_input=pv[1:2], return_dict=True)
            age = round(out.age_output[0][0].item())
            with self._lock:
                self._result = str(age)
            time.sleep(0.04)  # ~25 fps max — yield CPU to display loop


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_face_label(frame, x1, y1, x2, y2, confidence: float, age=None) -> None:
    """Draw the gold bounding box and HUMAN DETECTED badge with optional age."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    label = f"  HUMAN DETECTED  {confidence * 100:.1f}%"
    if age:
        label += f"  |  Age: {age}  "
    else:
        label += "  "
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    badge_y = y1 - th - 10 if y1 - th - 10 > 0 else y2 + 5
    cv2.rectangle(frame, (x1, badge_y), (x1 + tw, badge_y + th + baseline + 4), BADGE_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (x1, badge_y + th + 2), font, font_scale, TEXT_COLOR, thickness)


def draw_accessory_boxes(frame, accessories: list) -> None:
    """Draw a coloured bounding box and label for each detected accessory."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1

    for category, conf, x1, y1, x2, y2 in accessories:
        color = ACCESSORY_COLORS.get(category, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f" {category}  {conf * 100:.0f}% "
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Place label above the box; fall back to below if too close to top edge
        label_y = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(frame, (x1, label_y - th - 2), (x1 + tw, label_y + 3), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, label_y), font, font_scale, (0, 0, 0), thickness)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Limit PyTorch thread count so multiple background models don't fight over all CPU cores
    if TRANSFORMERS_AVAILABLE:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)

    # Validate face detection model files
    for path in (PROTOTXT, CAFFEMODEL):
        if not os.path.exists(path):
            print(f"Error: model file not found: {path}")
            sys.exit(1)

    # Stage 1: load SSD ResNet face detector (unchanged)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Stage 2: load YOLOv8 accessory detector (optional)
    acc_detector = None
    if not YOLO_AVAILABLE:
        print("ultralytics not installed — accessory detection disabled.")
        print("  Install:  pip install ultralytics")
    elif not os.path.exists(ACCESSORY_MODEL):
        print(f"Accessory model not found: {ACCESSORY_MODEL}")
        print("  Face detection will still run. See setup notes below.")
    else:
        print(f"Loading accessory model: {ACCESSORY_MODEL}")
        acc_detector = AccessoryDetector(ACCESSORY_MODEL)
        print("Accessory detection: ENABLED")

    # Stage 3: load age detector
    age_detector = None
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not installed — age detection disabled.")
    else:
        try:
            age_detector = AgeDetector()
            print("Age detection: ENABLED")
        except Exception as e:
            print(f"Age detector failed to load: {e}")

    # Stage 4: load watch detector (uses same ultralytics YOLO, auto-downloads model)
    watch_detector = None
    if not YOLO_AVAILABLE:
        print("ultralytics not installed — watch detection disabled.")
    else:
        try:
            watch_detector = WatchDetector()
            print("Watch detection: ENABLED")
        except Exception as e:
            print(f"Watch detector failed to load: {e}")

    # Stage 5: load fashion detector (YOLOS-Fashionpedia, hat/scarf/tie/bag/etc.)
    fashion_detector = None
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not installed — fashion detection disabled.")
    else:
        try:
            fashion_detector = FashionDetector()
            print("Fashion detection: ENABLED")
        except Exception as e:
            print(f"Fashion detector failed to load: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        print("On macOS, grant Terminal camera access in System Settings > Privacy & Security > Camera.")
        sys.exit(1)

    label = "Face + Accessory Detection" if acc_detector else "Face Detection"
    print(f"{label} running — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ---- Stage 1: face detection (logic unchanged) ----
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            # ---- Age prediction on face ROI (async) ----
            age_str = None
            if age_detector and (x2 - x1) >= ROI_MIN_PX:
                age_detector.submit(frame[y1:y2, x1:x2])
                age_str = age_detector.get_result()

            draw_face_label(frame, x1, y1, x2, y2, confidence, age=age_str)

            # ---- Stage 2: accessory detection on face ROI (async) ----
            if acc_detector and (x2 - x1) >= ROI_MIN_PX:
                # Pad the crop slightly so hats/glasses at the edge aren't clipped
                rx1 = max(0, x1 - ROI_PAD)
                ry1 = max(0, y1 - ROI_PAD)
                rx2 = min(w, x2 + ROI_PAD)
                ry2 = min(h, y2 + ROI_PAD)
                face_roi = frame[ry1:ry2, rx1:rx2]

                # Submit ROI + its frame offset to background thread (non-blocking)
                acc_detector.submit(face_roi, rx1, ry1)
                # Read last known results — always returns instantly
                accessories = acc_detector.get_results()
                draw_accessory_boxes(frame, accessories)

        # ---- Stage 3: watch detection on full frame (async) ----
        if watch_detector:
            watch_detector.submit(frame)
            draw_accessory_boxes(frame, watch_detector.get_results())

        # ---- Stage 5: fashion detection on full frame (async) ----
        if fashion_detector:
            fashion_detector.submit(frame)
            draw_accessory_boxes(frame, fashion_detector.get_results())

        cv2.imshow(label, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if age_detector:
        age_detector.stop()
    if acc_detector:
        acc_detector.stop()
    if watch_detector:
        watch_detector.stop()
    if fashion_detector:
        fashion_detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
