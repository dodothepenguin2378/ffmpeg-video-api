FROM python:3.11-slim

# Install FFmpeg and fonts
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create output directories
RUN mkdir -p /tmp/reel-output /tmp/reel-temp

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "660", "--workers", "1", "--threads", "2"]
