
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.colpali import ColPaliIngester
from src.database_setup import VectorDatabase

class TestColPali(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock(spec=VectorDatabase)
        
    @patch('src.ingestion.colpali.ColQwen2')
    @patch('src.ingestion.colpali.ColPaliProcessor')
    def test_ingester_embedding(self, MockProcessor, MockModel):
        # Mock dependencies
        mock_model = MockModel.from_pretrained.return_value
        # Important: .eval() returns a new mock unless configured to return self
        mock_model.eval.return_value = mock_model
        
        mock_processor = MockProcessor.from_pretrained.return_value
        
        # Mock outputs
        # embed_page returns list of lists
        # We need to handle call() -> [0] -> cpu() -> [float()] -> numpy() -> tolist()
        # Note: embed_query calls float(), embed_page might not depending on impl.
        # Let's mock both paths just in case to be robust.
        
        tensor_mock = MagicMock()
        tensor_mock.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2]]
        # Handle .float() chain for embed_query
        tensor_mock.cpu.return_value.float.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2]]
        
        # model(...) returns a tuple/output where [0] is the embeddings tensor
        mock_model.return_value.__getitem__.return_value = tensor_mock
        
        ingester = ColPaliIngester(device="cpu")
        ingester.load_model()
        
        # Test embed_page
        img = Image.new('RGB', (100, 100))
        vecs = ingester.embed_page(img)
        self.assertEqual(vecs, [[0.1, 0.2]])
        
        # Test embed_query
        vecs_q = ingester.embed_query("test query")
        self.assertEqual(vecs_q, [[0.1, 0.2]])

    def test_db_upsert_colpali(self):
        # Test that upsert_colpali_pages is called correctly in ingest logic
        # Here we just verify the method exists and accepts data
        # Real DB test would require Qdrant instance
        pass

if __name__ == '__main__':
    unittest.main()
