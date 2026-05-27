FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    openai \
    networkx \
    rapidfuzz \
    lingua-language-detector

COPY pipeline/ ./pipeline/

ENV VLLM_BASE_URL=http://localhost:8080/v1

ENTRYPOINT ["python3"]
CMD ["pipeline/extract_toponyms.py", "--help"]
