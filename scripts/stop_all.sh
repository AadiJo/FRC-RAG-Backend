#!/bin/bash
# Stop all screen sessions

echo "Stopping all FRC RAG services..."

screen -S api -X quit 2>/dev/null && echo "  ✓ API stopped" || echo "  - API not running"
screen -S tei -X quit 2>/dev/null && echo "  ✓ TEI stopped" || echo "  - TEI not running"
screen -S qdrant -X quit 2>/dev/null && echo "  ✓ Qdrant stopped" || echo "  - Qdrant not running"

echo ""
echo "All services stopped."
echo ""

# Check if any are still running
remaining=$(screen -ls 2>/dev/null | grep -E "qdrant|tei|api" || true)
if [ -n "$remaining" ]; then
    echo "WARNING: Some sessions still running:"
    echo "$remaining"
fi
