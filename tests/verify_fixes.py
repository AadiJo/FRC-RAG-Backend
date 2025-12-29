
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import tempfile
import shutil

# Add backend loop to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.captioner import ImageCaptioner, ProcessedImage
from src.query_processor import QueryProcessor
from src.utils.config import settings

class TestOptimizations(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_captioner_batching(self):
        """Verify ImageCaptioner calls batch method on vision model."""
        print("\nTesting Captioner Batching...")
        
        # Mock vision model
        mock_vision = MagicMock()
        mock_vision.describe_images_batch.return_value = ["Caption 1", "Caption 2"]
        
        captioner = ImageCaptioner(vision_model=mock_vision, device="cpu", use_ocr=False)
        
        # Create dummy processed images
        img1_path = self.test_dir / "img1.jpg"
        img2_path = self.test_dir / "img2.jpg"
        
        # Create dummy files so they exist
        from PIL import Image
        Image.new('RGB', (100, 100)).save(img1_path)
        Image.new('RGB', (100, 100)).save(img2_path)
        
        images = [
            ProcessedImage(image_id="img1", original_path=img1_path, saved_path=img1_path, page=1, width=100, height=100),
            ProcessedImage(image_id="img2", original_path=img2_path, saved_path=img2_path, page=2, width=100, height=100),
        ]
        
        # Call batch captioning
        captions = captioner.caption_processed_images(images, batch_size=2, show_progress=False)
        
        # Verify
        self.assertEqual(len(captions), 2)
        self.assertEqual(captions[0].raw_visual_facts, "Caption 1")
        self.assertEqual(captions[1].raw_visual_facts, "Caption 2")
        
        # Assert batch method was called
        mock_vision.describe_images_batch.assert_called_once()
        print("✓ ImageCaptioner used batch inference")

    @patch("src.query_processor.settings")
    def test_query_processor_strict_lookup(self, mock_settings):
        """Verify QueryProcessor strictly looks up images."""
        print("\nTesting QueryProcessor Strict Lookup...")
        
        # Setup fake images dir
        images_dir = self.test_dir / "images"
        images_dir.mkdir()
        mock_settings.images_path = str(images_dir)
        
        # Create a "real" image
        real_img = images_dir / "frc" / "9999" / "2024"
        real_img.mkdir(parents=True)
        (real_img / "real_id.jpg").touch()
        
        # Initialize QP
        # Mock DB to avoid loading real DB
        mock_db = MagicMock()
        with patch("src.query_processor.get_database", return_value=mock_db):
            qp = QueryProcessor(db=mock_db)
        
        # Check lookup
        url = qp._get_valid_image_url("real_id")
        print(f"Lookup 'real_id' -> {url}")
        
        self.assertIsNotNone(url)
        self.assertTrue(url.endswith("real_id.jpg"))
        
        # Check fake
        fake_url = qp._get_valid_image_url("made_up_id")
        print(f"Lookup 'made_up_id' -> {fake_url}")
        self.assertIsNone(fake_url)
        
        print("✓ QueryProcessor validated image IDs correctly")

if __name__ == "__main__":
    unittest.main()
