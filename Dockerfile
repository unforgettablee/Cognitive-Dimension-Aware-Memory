# MTL Pipeline Docker Image
#
# Usage:
#   docker build -t mtl-pipeline .
#   docker run -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY mtl-pipeline --help
#   docker run -v $(pwd)/memories:/app/memories -v $(pwd)/jobs:/app/jobs mtl-pipeline experiment -c harbor/configs/experiments/e1_full.yaml

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Memory Transfer Learning Pipeline"
LABEL org.opencontainers.image.description="Cross-domain memory transfer for coding agents with 4x4 cognitive matrix retrieval"
LABEL org.opencontainers.image.source="https://github.com/memorytransfer/mtl"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[harbor]"

# Pre-download sentence-transformers model (offline support)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY src/ src/
COPY harbor/config/ harbor/config/
COPY harbor/configs/ harbor/configs/

# Install the MTL package
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["mtl"]
CMD ["--help"]
