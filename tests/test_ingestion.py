"""
Unit tests for the ingestion modules.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDocumentParser:
    """Tests for DocumentParser."""

    def test_parse_filename(self):
        """Test filename parsing for team/year extraction."""
        from src.ingestion.parser import DocumentParser
        
        parser = DocumentParser()
        
        # Standard format
        team, year = parser._parse_filename("254-2025.pdf")
        assert team == "254"
        assert year == "2025"
        
        # Multi-part format
        team, year = parser._parse_filename("4607-1-2024.pdf")
        assert team == "4607"
        assert year == "2024"

    def test_extract_headers(self):
        """Test header extraction from text."""
        from src.ingestion.parser import DocumentParser
        
        parser = DocumentParser()
        
        text = """# Introduction
        
        Some content here.
        
        ## Design Overview
        
        More content.
        
        DRIVETRAIN SPECIFICATIONS
        
        Even more content.
        """
        
        headers = parser._extract_headers(text)
        
        assert "# Introduction" in headers or any("Introduction" in h for h in headers)
        assert "DRIVETRAIN SPECIFICATIONS" in headers

    def test_is_scanned_page_detection(self):
        """Test scanned page detection threshold."""
        from src.ingestion.parser import DocumentParser
        
        parser = DocumentParser()
        
        # The MIN_TEXT_DENSITY threshold is 50 characters
        assert parser.MIN_TEXT_DENSITY == 50


class TestChunker:
    """Tests for DocumentChunker."""

    def test_generate_chunk_id(self):
        """Test chunk ID generation format."""
        from src.ingestion.chunker import DocumentChunker
        
        chunker = DocumentChunker()
        
        chunk_id = chunker._generate_chunk_id("2025", "TestBinder", 5, 3)
        
        assert "2025" in chunk_id
        assert "p5" in chunk_id
        assert "s3" in chunk_id

    def test_detect_subsystem(self):
        """Test subsystem detection from text."""
        from src.ingestion.chunker import DocumentChunker
        
        chunker = DocumentChunker()
        
        # Test drivetrain detection
        text = "The drivetrain uses NEO motors with a 6 wheel configuration."
        headers = ["Drivetrain Design"]
        
        subsystem = chunker._detect_subsystem(text, headers)
        assert subsystem == "drivetrain"
        
        # Test intake detection
        text = "Our intake mechanism uses polycarbonate rollers to grab the game piece."
        headers = ["Intake System"]
        
        subsystem = chunker._detect_subsystem(text, headers)
        assert subsystem == "intake"

    def test_token_estimation(self):
        """Test token estimation function."""
        from src.ingestion.chunker import estimate_tokens
        
        # Rough estimate: 1 word ≈ 1.3 tokens
        text = "This is a test sentence with exactly ten words here."
        tokens = estimate_tokens(text)
        
        # 10 words * 1.3 ≈ 13 tokens
        assert 10 <= tokens <= 20


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_output_path_generation(self):
        """Test image output path structure."""
        from src.ingestion.image_processor import ImageProcessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ImageProcessor(output_dir=Path(tmpdir))
            
            path = processor._get_output_path(
                image_id="test_img_001",
                team="254",
                year="2025",
                format="png",
            )
            
            assert "frc" in str(path)
            assert "254" in str(path)
            assert "2025" in str(path)
            assert path.suffix == ".png"

    def test_format_selection_jpeg(self):
        """Test format selection for photos."""
        from src.ingestion.image_processor import ImageProcessor
        from PIL import Image
        import numpy as np
        
        processor = ImageProcessor()
        
        # Create a "photo-like" image with many colors
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")
        
        format = processor._select_format(img)
        assert format == "jpeg"

    def test_format_selection_png_transparency(self):
        """Test format selection for images with transparency."""
        from src.ingestion.image_processor import ImageProcessor
        from PIL import Image
        
        processor = ImageProcessor()
        
        # Create image with transparency
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        
        format = processor._select_format(img)
        assert format == "png"


class TestCaptioner:
    """Tests for ImageCaptioner."""

    def test_extract_numbers(self):
        """Test extraction of numbers from text."""
        from src.ingestion.captioner import ImageCaptioner
        
        # We need a mock vision model to avoid loading BLIP-2
        vision_model = MagicMock()
        captioner = ImageCaptioner(vision_model=vision_model, use_ocr=False)
        
        text = "The part #456 is 12.5mm long and weighs 3.2 lbs."
        numbers = captioner._extract_numbers(text)
        
        assert "456" in numbers
        assert "12.5" in numbers
        assert "3.2" in numbers

    def test_validate_caption_numbers(self):
        """Test caption grounding validation for numbers."""
        from src.ingestion.captioner import ImageCaptioner
        
        vision_model = MagicMock()
        captioner = ImageCaptioner(vision_model=vision_model, use_ocr=False)
        
        caption = "A 775pro motor with a 10:1 reduction."
        ocr_text = "775pro motor"
        context = "The intake uses a 10:1 gear ratio."
        
        passed, notes = captioner._validate_caption(caption, ocr_text, context)
        
        assert passed is True
        assert len(notes) == 0

    def test_validate_caption_hallucination(self):
        """Test detection of hallucinated numbers."""
        from src.ingestion.captioner import ImageCaptioner
        
        vision_model = MagicMock()
        captioner = ImageCaptioner(vision_model=vision_model, use_ocr=False)
        
        caption = "A robot with 6 wheels."
        ocr_text = "robot"
        context = "This is a prototype design."
        
        passed, notes = captioner._validate_caption(caption, ocr_text, context)
        
        # Validation should pass but show notes about ungrounded numbers
        assert passed is True
        assert any("ungrounded" in n.lower() for n in notes)
        assert "6" in notes[0]

class TestEmbedder:
    """Tests for embedding modules."""

    def test_embedding_result_to_dict(self):
        """Test EmbeddingResult serialization."""
        from src.ingestion.embedder import EmbeddingResult
        
        result = EmbeddingResult(
            id="test_chunk_001",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            model_version="1.0",
            embedding_dim=3,
            metadata={"team": "254"},
        )
        
        d = result.to_dict()
        
        assert d["id"] == "test_chunk_001"
        assert d["embedding"] == [0.1, 0.2, 0.3]
        assert d["model"] == "test-model"
        assert d["metadata"]["team"] == "254"


class TestConfig:
    """Tests for configuration module."""

    def test_default_settings(self):
        """Test default settings values."""
        from src.utils.config import Settings
        
        # Create settings without .env file
        settings = Settings(_env_file=None)
        
        assert settings.server_port == 5000
        assert settings.environment == "development"
        assert settings.api_key_required is False

    def test_api_keys_list(self):
        """Test API keys parsing."""
        from src.utils.config import Settings
        
        settings = Settings(valid_api_keys="key1,key2,key3", _env_file=None)
        
        keys = settings.api_keys_list
        
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_environment_detection(self):
        """Test environment mode detection."""
        from src.utils.config import Settings
        
        dev_settings = Settings(environment="development", _env_file=None)
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False
        
        prod_settings = Settings(environment="production", _env_file=None)
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True


class TestQueryProcessor:
    """Tests for QueryProcessor."""

    def test_normalize_query(self):
        """Test query normalization."""
        from src.query_processor import QueryProcessor
        
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.db = MagicMock()
        processor._text_embedder = None
        processor._image_embedder = None
        
        # Test whitespace stripping
        normalized = processor._normalize_query("  test query  ")
        assert normalized == "test query"

    def test_filter_low_confidence(self):
        """Test confidence filtering."""
        from src.query_processor import QueryProcessor, SearchResult
        
        processor = QueryProcessor.__new__(QueryProcessor)
        
        # Scenario: 4 results. Best is 0.9.
        # Ratio threshold is 0.5 * 0.9 = 0.45.
        # Absolute threshold is 0.3.
        results = [
            SearchResult(chunk_id="1", text="", score=0.9, page_number=1, team="", year="", binder=""),
            SearchResult(chunk_id="2", text="", score=0.7, page_number=1, team="", year="", binder=""),
            SearchResult(chunk_id="3", text="", score=0.4, page_number=1, team="", year="", binder=""),
            SearchResult(chunk_id="4", text="", score=0.2, page_number=1, team="", year="", binder=""),
        ]
        
        filtered = processor._filter_low_confidence(results, min_score=0.3)
        
        # Should keep first two (0.9, 0.7)
        # 0.4 is below ratio threshold (0.45)
        # 0.2 is below absolute (0.3)
        # Since len(filtered) < 3 and len(results) >= 3, it will trigger the "keep top few" rule.
        # In this implementation, results[:3] are returned.
        assert len(filtered) == 3 
        assert filtered[0].chunk_id == "1"
        assert filtered[2].chunk_id == "3"


class TestMetrics:
    """Tests for metrics module."""

    def test_counter_increment(self):
        """Test counter incrementing."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.increment("test_counter", 5)
        collector.increment("test_counter", 3)
        
        assert collector.get_counter("test_counter") == 8

    def test_ingestion_run_tracking(self):
        """Test ingestion run metrics."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.start_ingestion_run("test_run_001")
        collector.record_document_processed(success=True)
        collector.record_chunks_created(10)
        run = collector.end_ingestion_run(success=True)
        
        assert run is not None
        assert run.run_id == "test_run_001"
        assert run.documents_processed == 1
        assert run.chunks_created == 10

    def test_query_stats(self):
        """Test query statistics calculation."""
        from src.utils.metrics import metrics
        
        metrics.reset()
        
        # Record some queries
        metrics.record_query_result("q1", 5, 2, 100.0)
        metrics.record_query_result("q2", 3, 1, 150.0)
        metrics.record_query_result("q3", 7, 0, 80.0)
        
        stats = metrics.get_query_stats()
        
        assert stats["total_queries"] == 3
        assert stats["avg_latency_ms"] == 110.0


# Run with: pytest tests/test_ingestion.py -v
