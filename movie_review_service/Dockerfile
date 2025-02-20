FROM python:3.10.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Make startup script executable
RUN chmod +x start.sh

# Use startup script instead of direct uvicorn command
CMD ["/app/start.sh"] 