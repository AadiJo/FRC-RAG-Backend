# FRC RAG Backend

The high-performance RAG engine driving the FRC RAG platform - a Retrieval-Augmented Generation platform designed for FIRST Robotics Competition teams.

> **Note:** This repository contains the backend API server. For the frontend application, see [FRC-RAG-Frontend](https://github.com/AadiJo/FRC-RAG-Frontend).

## Overview

FRC teams often spend time creating new designs for mechanisms that other teams have already built successfully in past seasons. This happens because teams don't have an easy way to see what solutions already exist from previous competitions.

This project creates a searchable database that collects publicly shared technical documents from top-performing FRC teams across different seasons. The system uses RAG (Retrieval-Augmented Generation) technology to help users describe what they need and find relevant designs from past robots.

When teams search for solutions, the system looks through its collection to find mechanisms from earlier seasons that could work for current challenges. For example, a team looking for ways to pick up game pieces in the 2024 Reefscape season might discover that intake designs from the 2022 Rapid React cargo challenge could be modified to work with algae collection. The system shows pictures of actual robots and CAD drawings when available to help teams understand how the mechanisms work.

This tool helps FRC teams build on existing knowledge instead of starting from scratch, saving time and helping teams create better robots by learning from successful designs used by other teams.

## Features

- **Query Processing**: Enhanced RAG system with FRC game piece context mapping
- **Caching**: Semantic and exact-match caching for 30-90% faster responses
- **Rate Limiting**: Configurable rate limits to prevent abuse (default: 60 requests/minute)
- **Real-time Monitoring**: Performance monitoring and usage statistics with cache hit rates
- **REST API**: Full API for query processing, feedback, and health monitoring

## What makes FRC RAG different?

Users can upload documents, CAD files, and forum threads in real time, keeping the knowledge base up to date with current season innovations. The system uses multi-season mechanism retrieval and game-piece context mapping to return solutions that are truly relevant to the current challenge. With caching, rate limiting, and real-time monitoring, the platform is designed to be production-ready and deployable, so teams can integrate it into their workflow immediately

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenRouter API key:

```env
MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=your-api-key-here
```

### 3. Run the Server

```bash
./start.sh
# Or: python app.py
```

The API will be available at `http://localhost:5000`

## Project Structure

```
├── app.py                 # Main server application
├── requirements.txt       # Python dependencies
├── start.sh               # Server startup script
├── .env.example           # Environment configuration template
├── src/                   # Source code
│   ├── core/              # Core RAG components
│   │   ├── query_processor.py    # Query processing logic
│   │   ├── game_piece_mapper.py  # Game piece context mapping
│   │   ├── filter_config.py      # Filter configuration
│   │   └── image_embedder.py     # Image embedding utilities
│   ├── server/            # Server components
│   │   ├── config.py             # Configuration management
│   │   ├── rate_limiter.py       # Rate limiting
│   │   ├── ollama_proxy.py       # Ollama proxy
│   │   ├── openrouter_client.py  # OpenRouter client
│   │   └── tunnel.py             # Tunneling utilities
│   └── utils/             # Utilities
│       ├── database_setup.py     # Database initialization
│       ├── query_cache.py        # Query caching
│       └── feedback_manager.py   # Feedback handling
├── data/                  # Data directory (PDFs, images)
├── db/                    # ChromaDB database
├── logs/                  # Log files
├── scraper/               # Data scraping utilities
└── scripts/               # Utility scripts
```

## API Endpoints

### Query Endpoints
- `POST /api/query` - Process RAG query
- `POST /api/query/stream` - Stream RAG query response

### Health & Monitoring
- `GET /health` - Comprehensive health check
- `GET /api/stats` - Server statistics


### Utility Endpoints
- `POST /api/feedback` - Submit user feedback
- `GET /images/<path>` - Serve images
- `POST /api/upload/pdf` - Upload a PDF file

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PROVIDER` | LLM provider (`openrouter` or `local`) | `local` |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `SERVER_PORT` | Port to listen on | `5000` |
| `RATE_LIMIT_REQUESTS` | Requests per minute | `60` |
| `CHROMA_PATH` | ChromaDB path | `db` |
| `IMAGES_PATH` | Images directory | `data/images` |

## Planned Additions

* **Dynamic Source Integration**: Real-time document upload capability letting users to add team publications, Chief Delphi threads, and other relevant resources directly to the database during their current session
* **Visual Search Results**: Multi-modal query responses that combine text solution descriptions with corresponding robot photos and CAD screenshots for complete technical understanding
* **Mobile Application**: iOS and Android app providing full platform functionality
* **Shareable Collaboration**: Link generation system enabling teams to share specific search results and findings through persistent URLs for quick information sharing