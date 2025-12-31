#!/bin/bash
# Migrate local Qdrant DB to standalone Qdrant server via snapshots
# NO RE-INGESTION REQUIRED

set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "Qdrant Migration Tool (Snapshot-Based)"
echo "========================================"
echo ""

# Configuration
OLD_DB_PATH="${OLD_DB_PATH:-./db}"
NEW_QDRANT_URL="${NEW_QDRANT_URL:-http://127.0.0.1:6333}"

# Collections to migrate
COLLECTIONS=("frc_text_chunks" "frc_image_chunks" "frc_colpali" "user_docs")

# Check if old DB exists
if [ ! -d "$OLD_DB_PATH" ]; then
    echo "ERROR: Old database not found at $OLD_DB_PATH"
    exit 1
fi

echo "Source DB: $OLD_DB_PATH"
echo "Target Qdrant: $NEW_QDRANT_URL"
echo ""

# Check if new Qdrant is running
if ! curl -s "$NEW_QDRANT_URL/health" > /dev/null 2>&1; then
    echo "ERROR: New Qdrant server not responding at $NEW_QDRANT_URL"
    echo "Please start Qdrant first: ./scripts/start_qdrant.sh"
    exit 1
fi

echo "New Qdrant is healthy ✓"
echo ""

# Create backup
BACKUP_DIR="./backups/pre_migration_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r "$OLD_DB_PATH" "$BACKUP_DIR/"
cp -r data/images "$BACKUP_DIR/" 2>/dev/null || echo "  (no images to backup)"
echo "Backup created ✓"
echo ""

# Step 1: Start temporary Qdrant on old DB to create snapshots
echo "Step 1: Creating snapshots from old database..."
echo ""

# Find an available port for temp Qdrant
TEMP_PORT=6334
while nc -z localhost $TEMP_PORT 2>/dev/null; do
    TEMP_PORT=$((TEMP_PORT + 1))
done

echo "Starting temporary Qdrant on port $TEMP_PORT..."
qdrant --storage-path "$OLD_DB_PATH" &
TEMP_QDRANT_PID=$!
sleep 5

# Check if it started
if ! curl -s "http://127.0.0.1:$TEMP_PORT/health" > /dev/null 2>&1; then
    # Try default port
    TEMP_PORT=6333
    # The new Qdrant is on 6333, so we need to stop it temporarily
    echo "WARNING: Port conflict. Please stop new Qdrant and run this script again."
    kill $TEMP_QDRANT_PID 2>/dev/null || true
    exit 1
fi

echo "Temporary Qdrant running ✓"
echo ""

# Create snapshots
SNAPSHOT_DIR="./migration_snapshots_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAPSHOT_DIR"

for collection in "${COLLECTIONS[@]}"; do
    echo "Creating snapshot for $collection..."
    
    # Create snapshot
    response=$(curl -s -X POST "http://127.0.0.1:$TEMP_PORT/collections/$collection/snapshots")
    
    if echo "$response" | grep -q "error"; then
        echo "  WARNING: Failed to create snapshot for $collection"
        echo "  Response: $response"
        continue
    fi
    
    # Get snapshot name
    snapshot_name=$(echo "$response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    
    if [ -z "$snapshot_name" ]; then
        echo "  WARNING: Could not get snapshot name for $collection"
        continue
    fi
    
    echo "  Snapshot created: $snapshot_name"
    
    # Copy snapshot file
    snapshot_path="$OLD_DB_PATH/snapshots/$collection/$snapshot_name"
    if [ -f "$snapshot_path" ]; then
        cp "$snapshot_path" "$SNAPSHOT_DIR/"
        echo "  Copied to $SNAPSHOT_DIR/$snapshot_name ✓"
    fi
done

# Stop temporary Qdrant
echo ""
echo "Stopping temporary Qdrant..."
kill $TEMP_QDRANT_PID 2>/dev/null || true
wait $TEMP_QDRANT_PID 2>/dev/null || true
echo "Stopped ✓"
echo ""

# Step 2: Restore snapshots to new Qdrant
echo "Step 2: Restoring snapshots to new Qdrant..."
echo ""

for snapshot in "$SNAPSHOT_DIR"/*.snapshot; do
    if [ ! -f "$snapshot" ]; then
        continue
    fi
    
    snapshot_name=$(basename "$snapshot")
    # Extract collection name from snapshot filename
    collection=$(echo "$snapshot_name" | cut -d'-' -f1)
    
    echo "Restoring $snapshot_name to $collection..."
    
    # Get absolute path
    abs_path=$(realpath "$snapshot")
    
    response=$(curl -s -X PUT "$NEW_QDRANT_URL/collections/$collection/snapshots/recover" \
        -H 'Content-Type: application/json' \
        -d "{\"location\": \"file://$abs_path\"}")
    
    if echo "$response" | grep -q "error"; then
        echo "  ERROR: Failed to restore $collection"
        echo "  Response: $response"
    else
        echo "  Restored ✓"
    fi
done

echo ""
echo "========================================"
echo "Migration Complete!"
echo "========================================"
echo ""
echo "Verify collections:"
curl -s "$NEW_QDRANT_URL/collections" | python3 -m json.tool 2>/dev/null || curl -s "$NEW_QDRANT_URL/collections"
echo ""
echo ""
echo "Next steps:"
echo "  1. Update .env.local with:"
echo "     QDRANT_HOST=127.0.0.1"
echo "     QDRANT_PORT=6333"
echo "  2. Restart the API: screen -S api -X quit && ./scripts/start_api.sh"
echo "  3. Test: curl http://127.0.0.1:8000/api/v1/health"
echo ""
