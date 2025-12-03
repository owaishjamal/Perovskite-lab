FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch CPU-only first (smaller, faster)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Railway sets PORT env var, default to 8000)
EXPOSE 8000

# Start command - use startup script to handle PORT variable
CMD ["./start.sh"]

