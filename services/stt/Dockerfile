# Jetson-compatible base: L4T R36.x ships CUDA 12.6 on arm64.
# dustynv/faster-whisper is built for Jetson (arm64, JetPack 6.x).
# If the tag is unavailable, fall back to dustynv/cuda-python:r36.4.0 and
# install faster-whisper manually — but prefer the pre-built image.
FROM dustynv/faster-whisper:r36.4.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# faster-whisper caches models under /root/.cache/huggingface by default.
# Pre-download at build time so the first request isn't slow.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
