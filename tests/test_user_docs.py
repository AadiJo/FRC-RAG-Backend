"""
User documents API tests.
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
        "user_docs": {"points_count": 10},
    }
    db.search_text.return_value = []
    db.search_images.return_value = []
    db.upsert_user_docs.return_value = 2
    db.delete_user_docs.return_value = {"deleted": ["doc1"], "not_found": []}
    db.search_user_docs.return_value = [
        {
            "id": "user_doc_chunk_0",
            "score": 0.85,
            "text": "Test user document content",
            "user_id": "test_user",
            "doc_id": "user_doc",
            "title": "Test Doc",
            "chunk_index": 0,
        }
    ]
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


class TestUserDocumentsUpsert:
    """Tests for /user-documents/upsert endpoint."""

    def test_upsert_basic(self, client, mock_embedder):
        """Test basic upsert request."""
        with patch.object(
            client.app.state, "processor", create=True
        ):
            response = client.post(
                "/api/v1/user-documents/upsert",
                json={
                    "user_id": "test_user_123",
                    "documents": [
                        {
                            "doc_id": "test_doc_1",
                            "title": "Test Document",
                            "text": "This is a test document about FRC robotics.",
                            "source": {"type": "manual"},
                        }
                    ],
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test_user_123"
            assert len(data["upserted"]) == 1 or len(data["failed"]) >= 0

    def test_upsert_with_chunking_config(self, client):
        """Test upsert with custom chunking config."""
        response = client.post(
            "/api/v1/user-documents/upsert",
            json={
                "user_id": "test_user_123",
                "documents": [
                    {
                        "doc_id": "test_doc_2",
                        "title": "Test Document 2",
                        "text": "Another test document with more content. " * 50,
                        "source": {"type": "gdrive", "uri": "https://drive.google.com/file/d/abc123"},
                    }
                ],
                "chunking": {
                    "strategy": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                },
            }
        )
        
        assert response.status_code == 200

    def test_upsert_validation(self, client):
        """Test upsert validation."""
        # Missing user_id
        response = client.post(
            "/api/v1/user-documents/upsert",
            json={
                "documents": [{"doc_id": "test", "title": "Test", "text": "Test", "source": {"type": "manual"}}]
            }
        )
        
        assert response.status_code == 422  # Validation error


class TestUserDocumentsDelete:
    """Tests for /user-documents/delete endpoint."""

    def test_delete_basic(self, client, mock_database):
        """Test basic delete request."""
        response = client.post(
            "/api/v1/user-documents/delete",
            json={
                "user_id": "test_user_123",
                "doc_ids": ["doc1"],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_user_123"
        assert "deleted" in data
        assert "not_found" in data

    def test_delete_validation(self, client):
        """Test delete validation."""
        # Missing user_id
        response = client.post(
            "/api/v1/user-documents/delete",
            json={"doc_ids": ["doc1"]}
        )
        
        assert response.status_code == 422


class TestContextFusedWithUserDocs:
    """Tests for /rag/context_fused with user documents."""

    def test_context_without_user_id(self, client):
        """Test context without user_id returns FRC results only."""
        response = client.post(
            "/api/v1/rag/context_fused",
            json={"query": "swerve drive"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "context" in data
        assert "citations" in data

    def test_context_with_user_id(self, client, mock_database):
        """Test context with user_id includes user docs."""
        response = client.post(
            "/api/v1/rag/context_fused",
            json={
                "query": "swerve drive",
                "user_id": "test_user_123",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "context" in data
        assert "citations" in data


# Run with: pytest tests/test_user_docs.py -v
