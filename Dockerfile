FROM python:3.11-slim

# ffmpeg нужен для конвертации WAV -> MP3
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Cloud Run сам подставляет переменную PORT (по умолчанию 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "bot.py"]
