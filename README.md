# FRC RAG Backend

The high-performance RAG engine driving the FRC RAG platform - a Retrieval-Augmented Generation platform designed for FIRST Robotics Competition teams.

> **Note:** This repository contains the backend API server. For the frontend application, see [FRC-RAG-Frontend](https://github.com/AadiJo/FRC-RAG-Frontend).

## Overview

FRC teams often spend time creating new designs for mechanisms that other teams have already built successfully in past seasons. This happens because teams don't have an easy way to see what solutions already exist from previous competitions.

This project creates a searchable database that collects publicly shared technical documents from top-performing FRC teams across different seasons. The system uses RAG (Retrieval-Augmented Generation) technology to help users describe what they need and find relevant designs from past robots.

When teams search for solutions, the system looks through its collection to find mechanisms from earlier seasons that could work for current challenges. For example, a team looking for ways to pick up game pieces in the 2024 Reefscape season might discover that intake designs from the 2022 Rapid React cargo challenge could be modified to work with algae collection. The system shows pictures of actual robots and CAD drawings when available to help teams understand how the mechanisms work.

This tool helps FRC teams build on existing knowledge instead of starting from scratch, saving time and helping teams create better robots by learning from successful designs used by other teams.

## Features

- **Document Ingestion**: Parse PDF binders with OCR support
- **Image Processing**: Extract, deduplicate, and process images
- **Multimodal Embeddings**: Text (bge-large-en-v1.5) and image (CLIP ViT-L/14) embeddings
- **Vision Captioning**: Generate captions using BLIP-2 with context grounding
- **Vector Search**: Qdrant-powered hybrid search with late fusion
- **FastAPI Server**: Production-ready API with rate limiting and auth

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.local` and adjust settings:

```bash
ENVIRONMENT=development
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
DB_PATH=db
IMAGES_PATH=data/images
LOG_LEVEL=INFO
```

### 3. Ingest Documents

Place PDF files in `data/` directory, then run:

```bash
python scripts/ingest.py data/

# Options:
# --skip-images    Skip image processing (faster)
# --skip-captions  Skip caption generation
# --no-gpu         Disable GPU acceleration
# -v               Verbose output
```

### 4. Start Server

```bash
# Development
python -m src.app

# Or with uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 5000 --reload
```

### 5. Query the API

```bash
# Health check
curl http://localhost:5000/health

# Query
curl -X POST http://localhost:5000/context \
  -H "Content-Type: application/json" \
  -d '{"query": "How does a swerve drive work?"}'
```

## Project Structure

```
backend/
├── src/
│   ├── app.py              # FastAPI application
│   ├── database_setup.py   # Qdrant vector database
│   ├── query_processor.py  # Query and retrieval logic
│   ├── ingestion/          # Document processing
│   │   ├── parser.py       # PDF parsing with OCR
│   │   ├── image_processor.py  # Image extraction
│   │   ├── chunker.py      # Text chunking
│   │   ├── embedder.py     # Embedding generation
│   │   └── captioner.py    # Image captioning
│   └── utils/
│       ├── config.py       # Configuration management
│       ├── logger.py       # Structured logging
│       └── metrics.py      # Metrics collection
├── scripts/
│   └── ingest.py           # Ingestion pipeline CLI
├── tests/                  # Unit and integration tests
├── data/                   # PDF files and images
├── db/                     # Qdrant database
└── logs/                   # Application logs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Full search with all options |
| POST | `/context` | LLM-optimized context retrieval |
| POST | `/validate-citations` | Validate chunk IDs |
| GET | `/chunk/{chunk_id}` | Get specific chunk |
| GET | `/stats` | System statistics |
| GET | `/images/{path}` | Static image hosting |

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Environment mode |
| `SERVER_PORT` | 5000 | API port |
| `DB_PATH` | db | Qdrant database path |
| `IMAGES_PATH` | data/images | Image storage path |
| `API_KEY_REQUIRED` | false | Enable API key auth |
| `TUNNEL` | false | Enable ngrok tunnel |
| `LOG_LEVEL` | INFO | Logging level |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_api.py -v
```

## Architecture

### Ingestion Pipeline (GPU Machine)

1. **Parse**: Extract text and structure from PDFs
2. **OCR**: Process scanned pages with Tesseract/PaddleOCR
3. **Chunk**: Split into 400-600 token chunks
4. **Extract**: Save and deduplicate images
5. **Caption**: Generate image descriptions with BLIP-2
6. **Embed**: Create text (BGE) and image (CLIP) embeddings
7. **Export**: Save to Parquet for transfer

### Serving (VPS/Cloud)

1. **Ingest**: Load embeddings into Qdrant
2. **Query**: Embed queries on CPU with sentence-transformers
3. **Search**: Hybrid text + image vector search
4. **Fuse**: Late fusion scoring (0.7 text + 0.3 image)
5. **Format**: Return structured context for LLM

### Frontend Integration

The backend returns context only. The frontend:
1. Calls `/context` endpoint
2. Builds system prompt with context
3. Calls LLM API (Groq, OpenAI, etc.)
4. Displays response with citations

See `docs/frontend_integration.md` for detailed examples.

## Performance

- **Ingestion**: ~1-2 min per PDF (with GPU)
- **Query Latency**: <200ms (CPU embedding + search)
- **Memory**: ~2GB for 10k chunks