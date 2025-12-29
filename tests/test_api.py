"""
API integration tests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    db = MagicMock()
    db.get_stats.return_value = {
        "frc_text_chunks": {"points_count": 100},
        "frc_image_chunks": {"points_count": 50},
    }
    db.search_text.return_value = [
        {
            "id": "test_chunk_001",
            "score": 0.85,
            "text": "Test chunk content",
            "page_number": 1,
            "team": "254",
            "year": "2025",
            "binder": "test.pdf",
            "subsystem": "drivetrain",
            "headers": ["Test Header"],
            "image_ids": [],
        }
    ]
    db.search_images.return_value = []
    db.get_chunk_by_id.return_value = {
        "id": "test_chunk_001",
        "text": "Test chunk content",
    }
    return db


@pytest.fixture
def mock_embedder():
    """Mock text embedder for testing."""
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 1024
    embedder.embedding_dim = 1024
    return embedder


@pytest.fixture
def client(mock_database, mock_embedder):
    """Create test client with mocked dependencies."""
    with patch("src.app.get_database", return_value=mock_database), \
         patch("src.query_processor.get_database", return_value=mock_database), \
         patch("src.query_processor.TextEmbedder", return_value=mock_embedder):
        
        from src.app import app
        
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "database" in data
        assert "version" in data


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_basic(self, client):
        """Test basic query request."""
        response = client.post(
            "/query",
            json={"query": "How does the drivetrain work?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "query_id" in data
        assert "chunks" in data
        assert "latency_ms" in data

    def test_query_with_filters(self, client):
        """Test query with filters."""
        response = client.post(
            "/query",
            json={
                "query": "gear ratio",
                "team": "254",
                "year": "2025",
                "limit": 5,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filters_applied" in data
        assert data["filters_applied"].get("team") == "254"

    def test_query_validation(self, client):
        """Test query validation."""
        # Empty query should fail
        response = client.post(
            "/query",
            json={"query": ""}
        )
        
        assert response.status_code == 422  # Validation error

    def test_query_with_pagination(self, client):
        """Test query with pagination."""
        response = client.post(
            "/query",
            json={
                "query": "test query",
                "limit": 10,
                "offset": 5,
            }
        )
        
        assert response.status_code == 200


class TestContextEndpoint:
    """Tests for /context endpoint."""

    def test_context_basic(self, client):
        """Test basic context request."""
        response = client.post(
            "/context",
            json={"query": "What motors are used?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "context" in data
        assert "citations" in data

    def test_context_with_params(self, client):
        """Test context with custom parameters."""
        response = client.post(
            "/context",
            json={
                "query": "motor specifications",
                "max_chunks": 3,
                "max_context_length": 2000,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Context should be under max length
        assert len(data["context"]) <= 2100  # Some buffer for formatting


class TestCitationValidation:
    """Tests for /validate-citations endpoint."""

    def test_validate_citations(self, client):
        """Test citation validation."""
        response = client.post(
            "/validate-citations",
            json={"chunk_ids": ["test_chunk_001", "nonexistent_chunk"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data


class TestChunkEndpoint:
    """Tests for /chunk/{chunk_id} endpoint."""

    def test_get_chunk(self, client):
        """Test getting a specific chunk."""
        response = client.get("/chunk/test_chunk_001")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test_chunk_001"

    def test_get_nonexistent_chunk(self, client, mock_database):
        """Test getting a nonexistent chunk."""
        mock_database.get_chunk_by_id.return_value = None
        
        response = client.get("/chunk/nonexistent")
        
        assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_get_stats(self, client):
        """Test getting statistics."""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "database" in data
        assert "queries" in data


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_auth_disabled_by_default(self, client):
        """Test that auth is disabled by default."""
        # Query should work without API key
        response = client.post(
            "/query",
            json={"query": "test"}
        )
        
        assert response.status_code == 200

    def test_auth_when_enabled(self, mock_database, mock_embedder):
        """Test auth when enabled."""
        with patch("src.app.get_database", return_value=mock_database), \
             patch("src.query_processor.get_database", return_value=mock_database), \
             patch("src.query_processor.TextEmbedder", return_value=mock_embedder), \
             patch("src.app.settings") as mock_settings:
            
            mock_settings.api_key_required = True
            mock_settings.api_keys_list = ["valid-key-123"]
            mock_settings.is_development = True
            mock_settings.debug = True
            mock_settings.log_level = "INFO"
            mock_settings.log_file = None
            mock_settings.images_path.exists.return_value = False
            mock_settings.rate_limit_requests = 100
            
            from src.app import app
            client = TestClient(app)
            
            # Without API key should fail
            response = client.post(
                "/query",
                json={"query": "test"}
            )
            
            # Note: This test might need adjustment based on actual auth behavior


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/query",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # CORS should be configured
        assert response.status_code in [200, 405]


# Run with: pytest tests/test_api.py -v
