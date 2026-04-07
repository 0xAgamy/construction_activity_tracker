FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libpq-dev libsm6 libxext6 libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/ ./ui/
COPY services/cv_engine/ ./services/cv_engine/
COPY services/kafka_service/ ./services/kafka_service/

COPY models/ ./models/
COPY inputs/ ./inputs/

RUN mkdir -p outputs

EXPOSE 7860

CMD ["python", "ui/app.py"]