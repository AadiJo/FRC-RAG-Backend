# Migration Guide: Multi-Worker RAG Architecture

This guide covers migrating from the single-process embedded Qdrant setup to a multi-worker architecture with standalone Qdrant and Text Embeddings Inference (TEI).

## Overview

### Architecture Changes

| Component | Before | After |
|-----------|--------|-------|
| **Qdrant** | Embedded (local path) | Standalone server |
| **Embeddings** | In-process model | External TEI service |
| **API Workers** | 1 (uvicorn) | 3+ (uvicorn --workers) |
| **BM25** | Blocks event loop | ThreadPoolExecutor |
| **DB Queries** | Sync blocking | Async with backpressure |

### CPU Core Allocation (4-core VPS)

| Core | Service |
|------|---------|
| 0, 1 | FastAPI workers (3 workers) |
| 2 | TEI (embeddings) |
| 3 | Qdrant + Background worker |

---

## Prerequisites

### Install Required Binaries

**1. Qdrant**
```bash
# Download from releases
wget https://github.com/qdrant/qdrant/releases/download/v1.12.5/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/
```

**2. Text Embeddings Inference (TEI)**
```bash
# Option A: From HuggingFace releases
wget https://github.com/huggingface/text-embeddings-inference/releases/download/v1.5.0/text-embeddings-inference-cpu-1.5.0-x86_64-unknown-linux-gnu.tar.gz
tar -xzf text-embeddings-inference-cpu-1.5.0-x86_64-unknown-linux-gnu.tar.gz
sudo mv text-embeddings-router /usr/local/bin/text-embeddings-inference

# Option B: Docker (alternative)
# docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
```

**3. Screen**
```bash
sudo apt-get install screen
```

---

## Database Migration (Snapshot-Based, NO Re-Ingestion)

### Step 1: Backup Current Data

```bash
cd /home/aadi/L-Projects/frc-rag-again/backend

# Create timestamped backup
BACKUP_DIR=~/backup_$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
cp -r db/ $BACKUP_DIR/db
cp -r data/images/ $BACKUP_DIR/images
echo "Backup created at $BACKUP_DIR"
```

### Step 2: Create Snapshots from Existing DB

The existing `db/` folder contains embedded Qdrant data. To migrate:

```bash
# Start a temporary Qdrant server on your existing db
qdrant --storage-path ./db &
QDRANT_PID=$!
sleep 5

# Verify it's running
curl http://localhost:6333/collections

# Create snapshots for each collection
for col in frc_text_chunks frc_image_chunks frc_colpali user_docs; do
    echo "Creating snapshot for $col..."
    curl -X POST "http://localhost:6333/collections/$col/snapshots"
done

# List created snapshots
ls -la db/snapshots/*/

# Stop the temporary server
kill $QDRANT_PID
```

### Step 3: Start Standalone Qdrant with New Storage

```bash
# Create new storage directory
mkdir -p ~/qdrant_storage

# Start Qdrant with CPU pinning
screen -S qdrant
taskset -c 3 qdrant --storage-path ~/qdrant_storage

# Detach: Ctrl+A D
```

### Step 4: Restore Snapshots to New Qdrant

```bash
# Get the absolute path to your snapshot files
SNAPSHOT_DIR=$(pwd)/db/snapshots

# Restore each collection
for col in frc_text_chunks frc_image_chunks frc_colpali user_docs; do
    SNAPSHOT_FILE=$(ls $SNAPSHOT_DIR/$col/*.snapshot 2>/dev/null | head -1)
    if [ -n "$SNAPSHOT_FILE" ]; then
        echo "Restoring $col from $SNAPSHOT_FILE..."
        curl -X PUT "http://127.0.0.1:6333/collections/$col/snapshots/recover" \
            -H 'Content-Type: application/json' \
            -d "{\"location\": \"file://$SNAPSHOT_FILE\"}"
    else
        echo "No snapshot found for $col"
    fi
done
```

### Step 5: Verify Migration

```bash
# Check all collections exist with correct point counts
curl http://127.0.0.1:6333/collections

# Expected output should show:
# - frc_text_chunks: ~2946 points
# - frc_image_chunks: varies
# - frc_colpali: varies
# - user_docs: 0 points (unless you had user docs)
```

---

## Configuration

### Environment Variables

Create/update `.env.local` in the backend directory:

```bash
# Qdrant Configuration (REQUIRED for multi-worker)
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# TEI Configuration (REQUIRED for multi-worker)
TEI_URL=http://127.0.0.1:8080

# Concurrency Limits
MAX_CONCURRENT_EMBEDDINGS=10
MAX_CONCURRENT_QDRANT=20
MAX_CONCURRENT_BM25=4
BM25_THREAD_WORKERS=2

# API Configuration
API_WORKERS=3
API_PORT=8000

# Environment
ENVIRONMENT=production
DEBUG=false
```

### For Local Development (without TEI/remote Qdrant)

Leave `QDRANT_HOST` and `TEI_URL` unset to use:
- Embedded Qdrant (local `db/` folder)
- Local embedding model (falls back automatically)

---

## Starting the System

### Option A: Start All Services (Recommended)

```bash
cd /home/aadi/L-Projects/frc-rag-again/backend
./scripts/start_all.sh
```

### Option B: Start Services Individually

**Terminal 1 - Qdrant:**
```bash
screen -S qdrant
./scripts/start_qdrant.sh
# Ctrl+A D to detach
```

**Terminal 2 - TEI:**
```bash
screen -S tei
./scripts/start_tei.sh
# Ctrl+A D to detach
```

**Terminal 3 - API:**
```bash
screen -S api
./scripts/start_api.sh
# Ctrl+A D to detach
```

### Verify Services

```bash
# Check Qdrant
curl http://127.0.0.1:6333/health

# Check TEI (may take 30-60s for model loading)
curl http://127.0.0.1:8080/health

# Check API
curl http://127.0.0.1:8000/api/v1/health
```

---

## Operating the System

### Screen Commands

| Action | Command |
|--------|---------|
| List sessions | `screen -ls` |
| Attach to session | `screen -r api` |
| Detach from session | `Ctrl+A D` |
| Kill session | `screen -S api -X quit` |

### Stopping Services

```bash
./scripts/stop_all.sh
```

Or manually:
```bash
screen -S api -X quit
screen -S tei -X quit
screen -S qdrant -X quit
```

### Viewing Logs

```bash
# Real-time logs
tail -f logs/api.log
tail -f logs/tei.log
tail -f logs/qdrant.log

# Or attach to screen to see stdout
screen -r api
```

---

## Testing the Migration

### Quick Health Check

```bash
# Test query endpoint
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "swerve drive", "limit": 3}'
```

### Full Test Script

```python
import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health
        r = await client.get("http://127.0.0.1:8000/api/v1/health")
        print(f"Health: {r.json()}")
        
        # Test query
        r = await client.post(
            "http://127.0.0.1:8000/api/v1/query",
            json={"query": "drivetrain design", "limit": 3}
        )
        data = r.json()
        print(f"Query returned {len(data['chunks'])} chunks in {data['latency_ms']:.0f}ms")

asyncio.run(test())
```

---

## Troubleshooting

### Issue: "Storage folder is already accessed by another instance"

**Cause:** Multiple processes trying to access embedded Qdrant.

**Solution:** Set `QDRANT_HOST=127.0.0.1` to use remote Qdrant instead of embedded.

### Issue: TEI returns 503 or connection refused

**Cause:** TEI is still loading the model.

**Solution:** Wait 30-60 seconds for model loading. Check with:
```bash
curl http://127.0.0.1:8080/health
```

### Issue: High memory usage

**Cause:** Each API worker loads BM25 index (~200MB each).

**Solution:** This is expected. With 3 workers: ~600MB for BM25. Reduce `API_WORKERS` if needed.

### Issue: Slow first query

**Cause:** TEI/local embedder cold start.

**Solution:** First query loads the model. Subsequent queries are fast. TEI eliminates this per-worker.

---

## Performance Expectations

| Metric | Local Dev | Production (TEI + Qdrant) |
|--------|-----------|---------------------------|
| Cold start | ~20s | ~5s (pre-warmed) |
| Query latency | 2-3s | 200-500ms |
| Concurrent users | 1-2 | 10-20 |
| Memory usage | 4-6GB | 6-8GB total |

---

## Rollback

If migration fails, restore from backup:

```bash
# Stop all services
./scripts/stop_all.sh

# Restore database
rm -rf db/
cp -r $BACKUP_DIR/db ./db

# Remove remote config
# Edit .env.local and remove QDRANT_HOST and TEI_URL

# Restart in local mode
source .venv/bin/activate
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## Image Storage

Images remain at `data/images/` and are served via FastAPI's StaticFiles mount at `/images/*`. No migration needed.

If you need to move images to a different location:
1. Copy `data/images/` to new location
2. Update `IMAGES_PATH` in `.env.local`
3. Restart API

---

## Next Steps

1. **Set up monitoring:** Add Prometheus metrics endpoint
2. **Add log rotation:** Configure logrotate for `logs/*.log`
3. **Systemd services:** Convert screen scripts to systemd for auto-restart on reboot
4. **Load testing:** Use `locust` or `k6` to verify capacity
