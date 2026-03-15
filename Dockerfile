FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY eburon_tts_server.py .
COPY templates/ ./templates/

# Create output directory
RUN mkdir -p /tmp/eburon_tts_outputs

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "eburon_tts_server.py"]
