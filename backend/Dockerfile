FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip wget && rm -rf /var/lib/apt/lists/*

# llama.cpp
RUN wget -q https://github.com/ggml-org/llama.cpp/releases/download/b4467/llama-b4467-bin-ubuntu-x64.zip && \
    unzip llama-b4467-bin-ubuntu-x64.zip -d /llama && rm llama-b4467-bin-ubuntu-x64.zip

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/app/
WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
