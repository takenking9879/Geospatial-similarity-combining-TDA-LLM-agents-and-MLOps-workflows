# Dockerfile.flask — optimized for CPU-only PyTorch + LangChain

FROM python:3.11-slim-buster

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements_prod.txt /app/requirements_prod.txt

# Install uv (ultra-fast pip replacement)
RUN pip install --no-cache-dir uv

# Ensure uv uses CPU-only PyTorch wheels
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install CPU-only torch first (into system)
RUN uv pip install --system --index-strategy unsafe-best-match --prerelease=allow torch==2.9.0+cpu

# Install remaining dependencies (into system)
RUN uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements_prod.txt

# Copy all files of flask_app directly into /app/
COPY flask_app/ /app/

# Expose Gunicorn port
EXPOSE 8000

# Default command — note the corrected module path
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--workers", "3", "--threads", "2"]
