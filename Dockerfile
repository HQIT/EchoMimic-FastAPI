FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.fastapi.txt .
RUN pip install -r requirements.fastapi.txt

ENV FFMPEG_PATH=/app/ffmpeg-4.4-amd64-static

COPY . .

ENV LOG_LEVEL=debug

CMD ["sh", "-c", "uvicorn app:app --host ${SERVER_HOST} --port ${SERVER_PORT} --log-level ${LOG_LEVEL}"]