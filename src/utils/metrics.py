"""
Metrics collection module.

Tracks ingestion and retrieval metrics for monitoring and debugging.
All metrics are in-memory but can be exported to JSON for analysis.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimingMetric:
    """Single timing measurement."""
    
    name: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterMetric:
    """Counter metric with running total."""
    
    name: str
    count: int = 0
    last_updated: Optional[str] = None


@dataclass
class IngestionRun:
    """Metrics for a single ingestion run."""
    
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    images_extracted: int = 0
    images_deduplicated: int = 0
    embeddings_generated: int = 0
    total_duration_ms: float = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    model_info: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Thread-safe metrics collector.
    
    Tracks:
    - Ingestion metrics (documents, chunks, embeddings)
    - Retrieval metrics (latency, results, cache)
    - API metrics (requests, errors)
    """

    def __init__(self, export_path: Optional[Path] = None):
        """
        Initialize metrics collector.
        
        Args:
            export_path: Optional path for metrics export
        """
        self._lock = Lock()
        self._export_path = export_path
        
        # Ingestion metrics
        self._ingestion_runs: List[IngestionRun] = []
        self._current_ingestion: Optional[IngestionRun] = None
        
        # Retrieval metrics
        self._query_latencies: List[TimingMetric] = []
        self._query_results: List[Dict[str, Any]] = []
        
        # Counters
        self._counters: Dict[str, CounterMetric] = defaultdict(
            lambda: CounterMetric(name="")
        )
        
        # Rolling window metrics (last N entries)
        self._max_history = 1000

    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    # -------------------------------------------------------------------------
    # Ingestion Metrics
    # -------------------------------------------------------------------------

    def start_ingestion_run(
        self, run_id: str, model_info: Optional[Dict[str, str]] = None
    ) -> None:
        """Start tracking a new ingestion run."""
        with self._lock:
            self._current_ingestion = IngestionRun(
                run_id=run_id,
                started_at=self._now(),
                model_info=model_info or {},
            )
            logger.info(
                "Ingestion run started",
                run_id=run_id,
                model_info=model_info,
            )

    def end_ingestion_run(self, success: bool = True) -> Optional[IngestionRun]:
        """Complete the current ingestion run."""
        with self._lock:
            if not self._current_ingestion:
                return None
            
            self._current_ingestion.completed_at = self._now()
            
            # Calculate duration
            start = datetime.fromisoformat(
                self._current_ingestion.started_at.rstrip("Z")
            )
            end = datetime.fromisoformat(
                self._current_ingestion.completed_at.rstrip("Z")
            )
            self._current_ingestion.total_duration_ms = (
                end - start
            ).total_seconds() * 1000
            
            self._ingestion_runs.append(self._current_ingestion)
            result = self._current_ingestion
            self._current_ingestion = None
            
            logger.info(
                "Ingestion run completed",
                run_id=result.run_id,
                success=success,
                documents_processed=result.documents_processed,
                chunks_created=result.chunks_created,
                duration_ms=result.total_duration_ms,
            )
            
            return result

    def record_document_processed(self, success: bool = True) -> None:
        """Record a processed document."""
        with self._lock:
            if self._current_ingestion:
                if success:
                    self._current_ingestion.documents_processed += 1
                else:
                    self._current_ingestion.documents_failed += 1

    def record_chunks_created(self, count: int) -> None:
        """Record created chunks."""
        with self._lock:
            if self._current_ingestion:
                self._current_ingestion.chunks_created += count

    def record_images_extracted(self, count: int, deduplicated: int = 0) -> None:
        """Record extracted images."""
        with self._lock:
            if self._current_ingestion:
                self._current_ingestion.images_extracted += count
                self._current_ingestion.images_deduplicated += deduplicated

    def record_embeddings_generated(self, count: int) -> None:
        """Record generated embeddings."""
        with self._lock:
            if self._current_ingestion:
                self._current_ingestion.embeddings_generated += count

    def record_ingestion_error(
        self, error_type: str, message: str, document: Optional[str] = None
    ) -> None:
        """Record an ingestion error."""
        with self._lock:
            if self._current_ingestion:
                self._current_ingestion.errors.append({
                    "type": error_type,
                    "message": message,
                    "document": document,
                    "timestamp": self._now(),
                })
                logger.error(
                    "Ingestion error",
                    error_type=error_type,
                    message=message,
                    document=document,
                )

    # -------------------------------------------------------------------------
    # Retrieval Metrics
    # -------------------------------------------------------------------------

    @contextmanager
    def track_query(self, query_id: str, **metadata: Any):
        """
        Context manager to track query timing.
        
        Usage:
            with metrics.track_query("q123", user="test") as tracker:
                results = process_query()
                tracker.set_results(len(results))
        """
        start_time = time.perf_counter()
        result_count = 0
        
        class QueryTracker:
            def __init__(self, outer: "MetricsCollector"):
                self._outer = outer
                self.result_count = 0
                
            def set_results(self, count: int):
                self.result_count = count
        
        tracker = QueryTracker(self)
        
        try:
            yield tracker
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            with self._lock:
                metric = TimingMetric(
                    name="query",
                    duration_ms=duration_ms,
                    timestamp=self._now(),
                    metadata={
                        "query_id": query_id,
                        "result_count": tracker.result_count,
                        **metadata,
                    },
                )
                self._query_latencies.append(metric)
                
                # Trim history
                if len(self._query_latencies) > self._max_history:
                    self._query_latencies = self._query_latencies[-self._max_history:]
                
                logger.debug(
                    "Query completed",
                    query_id=query_id,
                    duration_ms=round(duration_ms, 2),
                    result_count=tracker.result_count,
                )

    def record_query_result(
        self,
        query_id: str,
        chunks_retrieved: int,
        images_retrieved: int,
        latency_ms: float,
    ) -> None:
        """Record query results directly."""
        with self._lock:
            # Record for history
            self._query_results.append({
                "query_id": query_id,
                "chunks_retrieved": chunks_retrieved,
                "images_retrieved": images_retrieved,
                "latency_ms": latency_ms,
                "timestamp": self._now(),
            })
            
            # Record as timing metric for statistics
            metric = TimingMetric(
                name="query",
                duration_ms=latency_ms,
                timestamp=self._now(),
                metadata={
                    "query_id": query_id,
                    "chunks_retrieved": chunks_retrieved,
                    "images_retrieved": images_retrieved,
                },
            )
            self._query_latencies.append(metric)
            
            # Trim history
            if len(self._query_results) > self._max_history:
                self._query_results = self._query_results[-self._max_history:]
            if len(self._query_latencies) > self._max_history:
                self._query_latencies = self._query_latencies[-self._max_history:]

    # -------------------------------------------------------------------------
    # Counter Metrics
    # -------------------------------------------------------------------------

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = CounterMetric(name=name)
            self._counters[name].count += amount
            self._counters[name].last_updated = self._now()

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, CounterMetric(name=name)).count

    # -------------------------------------------------------------------------
    # Statistics & Export
    # -------------------------------------------------------------------------

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        with self._lock:
            if not self._query_latencies:
                return {
                    "total_queries": 0,
                    "avg_latency_ms": 0,
                    "p50_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                }
            
            latencies = [m.duration_ms for m in self._query_latencies]
            latencies.sort()
            
            n = len(latencies)
            return {
                "total_queries": n,
                "avg_latency_ms": round(sum(latencies) / n, 2),
                "min_latency_ms": round(latencies[0], 2),
                "max_latency_ms": round(latencies[-1], 2),
                "p50_latency_ms": round(latencies[n // 2], 2),
                "p95_latency_ms": round(latencies[int(n * 0.95)], 2),
                "p99_latency_ms": round(latencies[int(n * 0.99)], 2),
            }

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        with self._lock:
            if not self._ingestion_runs:
                return {
                    "total_runs": 0,
                    "total_documents": 0,
                    "total_chunks": 0,
                    "total_images": 0,
                    "total_embeddings": 0,
                }
            
            return {
                "total_runs": len(self._ingestion_runs),
                "total_documents": sum(r.documents_processed for r in self._ingestion_runs),
                "failed_documents": sum(r.documents_failed for r in self._ingestion_runs),
                "total_chunks": sum(r.chunks_created for r in self._ingestion_runs),
                "total_images": sum(r.images_extracted for r in self._ingestion_runs),
                "total_embeddings": sum(r.embeddings_generated for r in self._ingestion_runs),
                "last_run": self._ingestion_runs[-1].run_id if self._ingestion_runs else None,
            }

    def export_metrics(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Export all metrics to JSON."""
        export_path = path or self._export_path
        
        with self._lock:
            metrics_data = {
                "exported_at": self._now(),
                "ingestion": {
                    "stats": self.get_ingestion_stats(),
                    "runs": [
                        {
                            "run_id": r.run_id,
                            "started_at": r.started_at,
                            "completed_at": r.completed_at,
                            "documents_processed": r.documents_processed,
                            "documents_failed": r.documents_failed,
                            "chunks_created": r.chunks_created,
                            "images_extracted": r.images_extracted,
                            "embeddings_generated": r.embeddings_generated,
                            "duration_ms": r.total_duration_ms,
                            "errors": r.errors,
                            "model_info": r.model_info,
                        }
                        for r in self._ingestion_runs
                    ],
                },
                "queries": {
                    "stats": self.get_query_stats(),
                    "recent": [
                        {
                            "timestamp": m.timestamp,
                            "duration_ms": m.duration_ms,
                            **m.metadata,
                        }
                        for m in self._query_latencies[-100:]
                    ],
                },
                "counters": {
                    name: {"count": c.count, "last_updated": c.last_updated}
                    for name, c in self._counters.items()
                },
            }
        
        if export_path:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            logger.info("Metrics exported", path=str(export_path))
        
        return metrics_data

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._ingestion_runs.clear()
            self._current_ingestion = None
            self._query_latencies.clear()
            self._query_results.clear()
            self._counters.clear()
            logger.info("Metrics reset")


# Global metrics instance
metrics = MetricsCollector()
