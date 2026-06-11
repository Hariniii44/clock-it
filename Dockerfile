FROM python:3.11-slim

WORKDIR /app

# System deps required by torch / tokenizers / spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spacy (not in requirements.txt) and download NER model
RUN pip install --no-cache-dir spacy && \
    python -m spacy download en_core_web_sm

# Pre-download HuggingFace models so cold starts are fast
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy source code
COPY config.py core_pipeline.py api.py ./
COPY src/ ./src/

# Copy only the data files needed at runtime (not datasets/ or indices/)
COPY data/bias_profiles.json \
     data/bias_profiles_era2.json \
     data/bias_profiles_era3.json \
     data/bias_matrix.json \
     data/bias_matrix_era2.json \
     data/bias_matrix_era3.json \
     data/article_groups.json \
     data/article_groups_era2.json \
     data/article_groups_era3.json \
     data/sri_lankan_politicians.json \
     data/sri_lankan_politicians_era2.json \
     data/sri_lankan_politicians_era3.json \
     data/bias_profiles_era_comparison.json \
     ./data/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
