FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/cv_engine/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy CV engine code
COPY services/cv_engine/ ./services/cv_engine/
COPY services/kafka_service/ ./services/kafka_service
COPY models/ ./models/
COPY inputs/ ./inputs/

# Create outputs directory
RUN mkdir -p outputs

CMD ["python", "services/cv_engine/main.py"]