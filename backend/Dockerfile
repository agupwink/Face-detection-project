FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights from project root
COPY res10_300x300_ssd_iter_140000.caffemodel .
COPY deploy.prototxt .
COPY accessories.pt .

# Copy face_detect.py so backend can import detector classes
COPY face_detect.py .

# Copy backend source
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
