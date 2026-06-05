# Jetson-compatible base: L4T R36.x ships CUDA 12.6 on arm64.
# dustynv/faster-whisper is built for Jetson (arm64, JetPack 6.x).
# If the tag is unavailable, fall back to dustynv/cuda-python:r36.4.0 and
# install faster-whisper manually — but prefer the pre-built image.
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
