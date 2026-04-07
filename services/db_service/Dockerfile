FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/cv_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/cv_engine/ ./services/cv_engine/
COPY models/ ./models/
COPY inputs/ ./inputs/

RUN mkdir -p outputs

CMD ["python", "services/cv_engine/main.py"]