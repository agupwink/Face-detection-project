import os
import sys
import time
import cv2
import numpy as np
import torch

# PyTorch 2.4+ defaults weights_only=True which breaks older ultralytics checkpoints.
# We trust our own model files so patching load to keep the pre-2.4 behaviour is safe.
_torch_load = torch.load
def _permissive_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load(f, *args, **kwargs)
torch.load = _permissive_load

# Allow importing face_detect.py from the project root when running locally
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detect import (
    AccessoryDetector, AgeDetector, WatchDetector, FashionDetector,
    PROTOTXT, CAFFEMODEL, ACCESSORY_MODEL,
    CONFIDENCE_THRESHOLD, ROI_MIN_PX, ROI_PAD,
    YOLO_AVAILABLE, TRANSFORMERS_AVAILABLE,
)


class DetectionPipeline:
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)

        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

        self.acc_detector = None
        if YOLO_AVAILABLE and os.path.exists(ACCESSORY_MODEL):
            self.acc_detector = AccessoryDetector(ACCESSORY_MODEL)

        self.age_detector = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.age_detector = AgeDetector()
            except Exception as e:
                print(f"Age detector failed: {e}")

        self.watch_detector = None
        if YOLO_AVAILABLE:
            try:
                self.watch_detector = WatchDetector()
            except Exception as e:
                print(f"Watch detector failed: {e}")

        self.fashion_detector = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.fashion_detector = FashionDetector()
            except Exception as e:
                print(f"Fashion detector failed: {e}")

        from trainer import load_bias
        self._age_bias: float = load_bias()
        if self._age_bias:
            print(f"[pipeline] Age bias correction loaded: {self._age_bias:+.1f} yrs")

        # TTL caches: expire stale results after 1.5 s so boxes don't linger
        self._watch_cache: list = []
        self._watch_expiry: float = 0.0
        self._fashion_cache: list = []
        self._fashion_expiry: float = 0.0
        self._acc_cache: list = []
        self._acc_expiry: float = 0.0

    def process_frame(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        raw = self.net.forward()

        faces = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1 = max(0, int(raw[0, 0, i, 3] * w))
            y1 = max(0, int(raw[0, 0, i, 4] * h))
            x2 = min(w, int(raw[0, 0, i, 5] * w))
            y2 = min(h, int(raw[0, 0, i, 6] * h))

            face_data = {
                "box": [x1, y1, x2, y2],
                "confidence": confidence,
                "age": None,
                "accessories": [],
            }

            if (x2 - x1) >= ROI_MIN_PX:
                face_roi = frame[y1:y2, x1:x2]

                if self.age_detector:
                    self.age_detector.submit(face_roi)
                    raw_age = self.age_detector.get_result()
                    if raw_age is not None and self._age_bias != 0.0:
                        corrected = max(1, round(int(raw_age) + self._age_bias))
                        raw_age = str(corrected)
                    face_data["age"] = raw_age

                if self.acc_detector:
                    rx1 = max(0, x1 - ROI_PAD)
                    ry1 = max(0, y1 - ROI_PAD)
                    rx2 = min(w, x2 + ROI_PAD)
                    ry2 = min(h, y2 + ROI_PAD)
                    self.acc_detector.submit(frame[ry1:ry2, rx1:rx2], rx1, ry1)
                    fresh = self.acc_detector.get_results()
                    if fresh:
                        self._acc_cache = [
                            {"label": cat, "confidence": conf, "box": [bx1, by1, bx2, by2]}
                            for cat, conf, bx1, by1, bx2, by2 in fresh
                        ]
                        self._acc_expiry = time.time() + 1.5
                    elif time.time() > self._acc_expiry:
                        self._acc_cache = []
                    face_data["accessories"] = self._acc_cache

            faces.append(face_data)

        now = time.time()

        watches = []
        if self.watch_detector:
            self.watch_detector.submit(frame)
            fresh = self.watch_detector.get_results()
            if fresh:
                self._watch_cache = [
                    {"label": label, "confidence": conf, "box": [x1, y1, x2, y2]}
                    for label, conf, x1, y1, x2, y2 in fresh
                ]
                self._watch_expiry = now + 1.5
            elif now > self._watch_expiry:
                self._watch_cache = []
            watches = self._watch_cache

        fashion = []
        if self.fashion_detector:
            self.fashion_detector.submit(frame)
            fresh = self.fashion_detector.get_results()
            if fresh:
                self._fashion_cache = [
                    {"label": label, "confidence": conf, "box": [x1, y1, x2, y2]}
                    for label, conf, x1, y1, x2, y2 in fresh
                ]
                self._fashion_expiry = now + 1.5
            elif now > self._fashion_expiry:
                self._fashion_cache = []
            fashion = self._fashion_cache

        # If no face in frame, clear all caches immediately
        if not faces:
            self._watch_cache = []
            self._fashion_cache = []
            self._acc_cache = []
            self._watch_expiry = 0.0
            self._fashion_expiry = 0.0
            self._acc_expiry = 0.0

        return {"faces": faces, "watches": watches, "fashion": fashion}

    def reset_for_new_session(self) -> None:
        if self.age_detector:
            self.age_detector.reset()
        self._watch_cache = []
        self._fashion_cache = []
        self._acc_cache = []
        self._watch_expiry = 0.0
        self._fashion_expiry = 0.0
        self._acc_expiry = 0.0

    def reload_bias(self) -> None:
        from trainer import load_bias
        self._age_bias = load_bias()
        print(f"[pipeline] Age bias reloaded: {self._age_bias:+.1f} yrs")

    def shutdown(self):
        if self.age_detector:
            self.age_detector.stop()
        if self.acc_detector:
            self.acc_detector.stop()
        if self.watch_detector:
            self.watch_detector.stop()
        if self.fashion_detector:
            self.fashion_detector.stop()
