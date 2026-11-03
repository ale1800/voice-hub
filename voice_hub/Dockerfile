FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install wyoming==1.10.2 sherpa-onnx==1.13.8 onnxruntime==1.31.0 numpy aiohttp
# Speaker embedding model: NVIDIA NeMo TitaNet large (CC-BY-4.0), ~113 MB, as
# exported for sherpa-onnx. See NOTICE.md.
RUN mkdir -p /models && curl -fsSL -o /models/nemo_en_titanet_large.onnx \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx
ENV VOICE_HUB_MODEL=/models/nemo_en_titanet_large.onnx VOICE_HUB_DATA=/share/voice_hub
WORKDIR /app
COPY app/ /app/
# openWakeWord features + the hey_jarvis head, for the second-stage wake word (see models/oww/NOTICE.md).
COPY models/oww/ /models/oww/
COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
