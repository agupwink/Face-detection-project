import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

try:
    from facenet_pytorch import InceptionResnetV1
    _resnet = InceptionResnetV1(pretrained='vggface2').eval()
    FACENET_AVAILABLE = True
except Exception as e:
    FACENET_AVAILABLE = False
    print(f"facenet-pytorch not available — face embeddings disabled: {e}")

_transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_face_embedding(face_bgr: np.ndarray) -> list[float] | None:
    if not FACENET_AVAILABLE or face_bgr is None or face_bgr.size == 0:
        return None
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = _transform(Image.fromarray(face_rgb)).unsqueeze(0)
        with torch.no_grad():
            emb = _resnet(tensor)
        return emb.squeeze().numpy().tolist()
    except Exception:
        return None
