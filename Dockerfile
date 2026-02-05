# Python backend for bison simulation
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY requirements.txt .

# Create data directory and copy GeoTIFF
# Note: The GeoTIFF should be copied to python_backend/data/ before building
RUN mkdir -p /app/data
COPY data/ /app/data/

# Cloud Run sets PORT environment variable
ENV PORT=8080

# Run the server
CMD exec uvicorn server:app --host 0.0.0.0 --port $PORT
