FROM python:3.13-slim

WORKDIR /workspace

# Install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Generate synthetic dataset and train models to serialize artifacts during the build phase
RUN .venv/bin/python ml/generate_data.py && \
    .venv/bin/python ml/train.py

# Expose the default port
EXPOSE 8000

# Use dynamic shell variable for PORT compatibility with Render/Railway
CMD ["sh", "-c", ".venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
