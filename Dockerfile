FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV, ffmpeg, PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runtime directories
RUN mkdir -p \
    uploads/backend \
    uploads/frontend \
    uploads/speech \
    uploads/skin_lesion_output \
    uploads/brain_tumor_output \
    data/qdrant_db \
    data/docs_db \
    data/parsed_docs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
