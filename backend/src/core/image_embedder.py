import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os

class ImageEmbedder:
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP model {model_name} on {self.device}...")
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def embed_image(self, image):
        """
        Embed an image or list of images.
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            List of embeddings (list of floats)
        """
        try:
            # Handle single image
            if isinstance(image, Image.Image):
                images = [image]
            else:
                images = image

            # Ensure images are RGB
            images = [img.convert('RGB') for img in images]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embeddings (SigLIP/CLIP usually benefit from this, though get_image_features might not be normalized by default)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error embedding image: {e}")
            return []

    def embed_text(self, text):
        """
        Embed a text string or list of strings.
        Args:
            text: String or list of strings
        Returns:
            List of embeddings (list of floats)
        """
        try:
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text

            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            return text_features.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []
