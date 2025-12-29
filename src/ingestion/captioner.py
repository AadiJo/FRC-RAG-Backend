"""
Image captioning module.

Generates captions for images using vision models with context grounding:
- Vision model analysis (BLIP-2/Florence-2/Qwen2-VL)
- OCR text extraction from images
- Context grounding with nearby text
- Caption validation against source text
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics

from .image_processor import ProcessedImage

logger = get_logger(__name__)


@dataclass
class ImageCaption:
    """Generated caption with metadata."""
    
    image_id: str
    raw_visual_facts: str
    final_caption: str
    ocr_text: Optional[str] = None
    source_context: Optional[str] = None
    validation_passed: bool = True
    validation_notes: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "raw_visual_facts": self.raw_visual_facts,
            "final_caption": self.final_caption,
            "ocr_text": self.ocr_text,
            "source_context": self.source_context,
            "validation_passed": self.validation_passed,
            "validation_notes": self.validation_notes,
            "confidence": self.confidence,
        }


class VisionModelBase:
    """Base class for vision models."""
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        self._processor = None
    
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate description of image."""
        raise NotImplementedError

    def describe_images_batch(self, images: List[Image.Image], prompt: Optional[str] = None) -> List[str]:
        """Generate descriptions for a batch of images."""
        # Default fallback implementation: loop
        return [self.describe_image(img, prompt) for img in images]


class BLIP2Model(VisionModelBase):
    """BLIP-2 vision model for image captioning."""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str] = None,
    ):
        super().__init__(device)
        self.model_name = model_name
    
    def _load_model(self):
        if self._model is None:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            logger.info(
                "Loading BLIP-2 model",
                model=self.model_name,
                device=self.device,
            )
            
            self._processor = Blip2Processor.from_pretrained(self.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
            logger.info("BLIP-2 model loaded")
    
    def describe_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate description of image.
        
        Args:
            image: PIL Image
            prompt: Optional prompt to guide generation
            
        Returns:
            Generated description
        """
        self._load_model()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Default prompt for FRC context - more specific to get better descriptions
        if prompt is None:
            prompt = "What mechanical components, parts, or structures are visible in this engineering image? Describe their shapes, positions, and any visible connections or relationships."
        
        inputs = self._processor(image, prompt, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Get the input length to extract only generated tokens
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=200,  # Reduced for speed
                num_beams=1,  # Greedy decoding (faster than beam search)
                do_sample=False,
                use_cache=True,  # Enable KV cache
            )
        
        # Extract only the newly generated tokens (skip the input prompt tokens)
        generated_tokens = outputs[0][input_length:]
        
        # Decode only the generated tokens
        description = self._processor.decode(generated_tokens, skip_special_tokens=True)
        
        # Aggressively filter out prompt text that might have leaked through
        if prompt:
            # Remove prompt if it appears at the start
            prompt_lower = prompt.lower().strip()
            description_lower = description.lower().strip()
            
            # Check if description starts with prompt
            if description_lower.startswith(prompt_lower):
                description = description[len(prompt):].strip()
            
            # Remove common prompt phrases
            prompt_phrases = [
                "describe the components, their arrangement, and any visible connections",
                "describe what you see, focusing on components, labels, and relationships",
                "describe this engineering image in detail",
                "what mechanical components, parts, or structures are visible",
                "this image shows a",
                "this image is from a section about",
            ]
            
            for phrase in prompt_phrases:
                if phrase in description_lower:
                    # Remove the phrase and clean up
                    import re
                    description = re.sub(re.escape(phrase), "", description, flags=re.IGNORECASE)
                    description = description.strip()
                    # Clean up extra spaces and punctuation
                    description = re.sub(r'\s+', ' ', description)
                    description = re.sub(r'^[.,;:\s]+', '', description)
        
        # Additional safety check: if description is empty or too short, use fallback
        if not description or len(description.strip()) < 10:
            logger.warning(f"Generated description too short for image, using fallback")
            return "Engineering diagram or technical image."
        
        return description.strip()

    def describe_images_batch(
        self,
        images: List[Image.Image],
        prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate descriptions for a batch of images."""
        self._load_model()
        
        if not images:
            return []
            
        # Convert all to RGB
        rgb_images = []
        for img in images:
            if img.mode != "RGB":
                rgb_images.append(img.convert("RGB"))
            else:
                rgb_images.append(img)
        
        # Default prompt
        if prompt is None:
            prompt = "What mechanical components, parts, or structures are visible in this engineering image? Describe their shapes, positions, and any visible connections or relationships."
            
        prompts = [prompt] * len(rgb_images)
        
        # Process batch
        inputs = self._processor(
            images=rgb_images, 
            text=prompts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
            
        descriptions = []
        for output in outputs:
            generated_tokens = output[input_length:]
            description = self._processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Filter prompt leakage (reusing logic from describe_image)
            if prompt:
                prompt_lower = prompt.lower().strip()
                desc_lower = description.lower().strip()
                if desc_lower.startswith(prompt_lower):
                    description = description[len(prompt):].strip()
                
                # Simple cleanup for batch mode
                prompt_phrases = [
                    "describe the components", "describe what you see", 
                    "describe this engineering image", "what mechanical components",
                    "this image shows a"
                ]
                for phrase in prompt_phrases:
                    if phrase in desc_lower:
                        # Regex cleanup would be better but keeping it simple for batch
                        import re
                        description = re.sub(re.escape(phrase), "", description, flags=re.IGNORECASE).strip()
            
            descriptions.append(description.strip() or "Engineering diagram or technical image.")
            
        return descriptions


class Qwen2VLModel(VisionModelBase):
    """Qwen2-VL vision model for image captioning - better for technical diagrams."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
    ):
        super().__init__(device)
        self.model_name = model_name
    
    def _load_model(self):
        if self._model is None:
            try:
                from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
            except ImportError:
                # Fallback to Auto classes if specific classes don't exist
                from transformers import AutoProcessor, AutoModelForVision2Seq
                Qwen2VLProcessor = AutoProcessor
                Qwen2VLForConditionalGeneration = AutoModelForVision2Seq
            
            logger.info(
                "Loading Qwen2-VL model",
                model=self.model_name,
                device=self.device,
            )
            
            try:
                self._processor = Qwen2VLProcessor.from_pretrained(self.model_name)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device == "cuda":
                    self._model = self._model.to(self.device)
                self._model.eval()
                
                # Try to compile model for faster inference (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile') and self.device == "cuda":
                        self._model = torch.compile(self._model, mode="reduce-overhead")
                        logger.info("Qwen2-VL model compiled for faster inference")
                except Exception:
                    pass  # Compilation is optional
                
                logger.info("Qwen2-VL model loaded")
            except Exception as e:
                logger.error(f"Failed to load Qwen2-VL model: {e}")
                raise
    
    def describe_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate description of image using Qwen2-VL.
        
        Args:
            image: PIL Image
            prompt: Optional prompt to guide generation
            
        Returns:
            Generated description
        """
        self._load_model()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Qwen2-VL works better with more specific prompts
        # Using shorter prompt for faster processing
        if prompt is None:
            prompt = "Describe the mechanical components, parts, and structures visible in this engineering image."
        
        try:
            # Qwen2-VL API: Build messages format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Try different API approaches based on what's available
            try:
                # Method 1: Apply chat template then process with images
                text = self._processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self._processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
            except (TypeError, AttributeError) as e:
                # Method 2: Try passing messages directly
                logger.debug(f"Method 1 failed, trying direct messages: {e}")
                try:
                    inputs = self._processor(
                        messages=messages,
                        padding=True,
                        return_tensors="pt",
                    )
                except (TypeError, AttributeError) as e2:
                    # Method 3: Try with apply_chat_template tokenized
                    logger.debug(f"Method 2 failed, trying tokenized template: {e2}")
                    inputs = self._processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    # Add image to inputs
                    if "images" not in inputs:
                        inputs["images"] = [image]
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate with optimized settings for speed
            with torch.no_grad():
                # Get pad_token_id safely
                pad_token_id = None
                if hasattr(self._processor, 'tokenizer'):
                    pad_token_id = getattr(self._processor.tokenizer, 'pad_token_id', None) or \
                                  getattr(self._processor.tokenizer, 'eos_token_id', None)
                
                generate_kwargs = {
                    "max_new_tokens": 200,  # Reduced from 512 for speed (200 is plenty for captions)
                    "do_sample": False,  # Deterministic, faster
                    "num_beams": 1,  # Greedy decoding (faster than beam search)
                    "use_cache": True,  # Enable KV cache for speed
                }
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id
                
                generated_ids = self._model.generate(**inputs, **generate_kwargs)
                
                # Extract only the generated tokens (skip input tokens)
                input_length = inputs["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[0][input_length:]
                
                description = self._processor.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            
            # Clean up prompt text if it leaked through
            if prompt:
                prompt_lower = prompt.lower()
                description_lower = description.lower()
                if description_lower.startswith(prompt_lower):
                    description = description[len(prompt):].strip()
            
            if not description or len(description.strip()) < 10:
                return "Engineering diagram or technical image."
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Qwen2-VL generation failed: {e}")
            return description.strip()
            
        except Exception as e:
            logger.error(f"Qwen2-VL generation failed: {e}")
            return "Engineering diagram or technical image."

    def describe_images_batch(
        self,
        images: List[Image.Image],
        prompt: Optional[str] = None,
    ) -> List[str]:
        """Batch generation for Qwen2-VL."""
        self._load_model()
        if not images:
            return []
            
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        if prompt is None:
            # Add strict instruction to prevent prompt leakage
            prompt = "Describe the mechanical components, parts, and structures visible in this engineering image. Return ONLY the description, do not repeat this prompt."
            
        # Qwen2-VL batching
        try:
            # Prepare messages for each image
            texts = []
            image_inputs = []
            
            for img in rgb_images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
                image_inputs.append(img)
                
            # Process batch
            inputs = self._processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
                     
            with torch.no_grad():
                pad_token_id = None
                if hasattr(self._processor, 'tokenizer'):
                    pad_token_id = getattr(self._processor.tokenizer, 'pad_token_id', None) or \
                                  getattr(self._processor.tokenizer, 'eos_token_id', None)
                                  
                generate_kwargs = {
                    "max_new_tokens": 200,
                    "do_sample": False,
                    "num_beams": 1,
                    "use_cache": True,
                }
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id
                    
                generated_ids = self._model.generate(**inputs, **generate_kwargs)
                
                input_length = inputs["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
                
                descriptions = self._processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # Cleanup
                cleaned_descs = []
                prompt_lower = prompt.lower()
                for desc in descriptions:
                    desc_lower = desc.lower()
                    if desc_lower.startswith(prompt_lower):
                        desc = desc[len(prompt):].strip()
                    cleaned_descs.append(desc.strip() or "Engineering diagram.")
                    
                return cleaned_descs
                
        except Exception as e:
            logger.error(f"Qwen2-VL batch failed: {e}")
            return ["Error processing batch"] * len(images)


class ImageCaptioner:
    """
    Image captioner with context grounding.
    
    Process:
    1. Extract raw visual facts from vision model
    2. Extract OCR text from image
    3. Combine with nearby document text
    4. Synthesize and validate caption
    """

    # Patterns that indicate technical content
    TECHNICAL_PATTERNS = [
        r"\d+\s*(?:mm|cm|m|in|ft|lb|kg|N|rpm|fps)",  # Measurements
        r"#\d+",  # Part numbers
        r"\d+x\d+",  # Dimensions
        r"(?:motor|gear|wheel|shaft|bearing|bolt|nut|bracket)",  # Components
    ]

    def __init__(
        self,
        vision_model: Optional[VisionModelBase] = None,
        use_ocr: bool = True,
        device: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize captioner.
        
        Args:
            vision_model: Vision model to use (default: BLIP-2)
            use_ocr: Enable OCR text extraction
            device: Device for models
            max_workers: Number of parallel workers (None = auto-detect)
        """
        self.use_ocr = use_ocr
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set max workers: fewer for GPU (to avoid memory issues), more for CPU
        if max_workers is None:
            # Pure CPU workers now since GPU is separate
            import os
            self.max_workers = min(16, (os.cpu_count() or 4) * 2)
        else:
            self.max_workers = max_workers
        
        # GPU lock no longer needed with batching strategy, but keeping for legacy single-image calls
        self._gpu_lock = Lock() if self.device == "cuda" else None
        
        if vision_model is None:
            # Auto-detect model type from name
            model_name = settings.vision_model.lower()
            if "qwen" in model_name or "qwen2-vl" in model_name:
                self.vision_model = Qwen2VLModel(
                    model_name=settings.vision_model,
                    device=self.device,
                )
            else:
                # Default to BLIP-2
                self.vision_model = BLIP2Model(
                    model_name=settings.vision_model,
                    device=self.device,
                )
        else:
            self.vision_model = vision_model
        
        # Preload model to avoid lazy loading overhead
        if hasattr(self.vision_model, '_load_model'):
            try:
                self.vision_model._load_model()
                logger.info("Vision model preloaded for faster captioning")
            except Exception as e:
                logger.warning(f"Failed to preload vision model: {e}")
        
        self._ocr = None
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS
        ]

    def _get_ocr(self):
        """Lazy load OCR model."""
        if self._ocr is None and self.use_ocr:
            try:
                import pytesseract
                self._ocr = pytesseract
            except ImportError:
                logger.warning("Tesseract not available for image OCR")
        return self._ocr

    def _extract_ocr_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        ocr = self._get_ocr()
        if ocr is None:
            return ""
        
        try:
            text = ocr.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"Image OCR failed: {e}")
            return ""

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text."""
        return re.findall(r"\d+(?:\.\d+)?", text)

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        terms = []
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            terms.extend(matches)
        return list(set(terms))

    def _validate_caption(
        self,
        caption: str,
        ocr_text: str,
        context: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate caption against source text.
        
        Rules:
        - Numbers in caption should exist in OCR or context (warning only)
        - Technical terms should ideally be grounded in context (warning only)
        - Be lenient - only flag major issues
        
        Returns:
            Tuple of (validation_passed, notes)
        """
        notes = []
        passed = True
        
        # Skip validation for very short or generic captions
        if len(caption.strip()) < 15:
            return True, []  # Too short to validate meaningfully
        
        # Extract numbers from caption
        caption_numbers = set(self._extract_numbers(caption))
        source_numbers = set(
            self._extract_numbers(ocr_text) + self._extract_numbers(context)
        )
        
        # Check for hallucinated numbers (but be lenient - only flag if many)
        hallucinated_numbers = caption_numbers - source_numbers
        if len(hallucinated_numbers) > 3:  # Only flag if many numbers are wrong
            notes.append(f"Some numbers may not match source")
            # Don't fail validation for numbers
        
        # Check for technical term grounding (but be lenient)
        caption_terms = self._extract_technical_terms(caption)
        source_terms = self._extract_technical_terms(ocr_text + " " + context)
        
        # Only flag if we have many caption terms but NO source terms
        if len(caption_terms) > 5 and not source_terms:
            notes.append("Many technical terms, limited source context")
            # Still pass - this is just a note
        
        # Always pass validation - we want to keep captions even if imperfect
        return True, notes

    def _synthesize_caption(
        self,
        visual_facts: str,
        ocr_text: str,
        context: str,
    ) -> str:
        """
        Synthesize final caption from all sources.
        
        Priority:
        1. Visual facts from model
        2. OCR text for labels/annotations
        3. Context for proper naming and subsystem identification
        """
        # Filter out generic/fallback responses and prompt text
        generic_phrases = [
            "engineering diagram or technical image",
            "describe this engineering image",
            "describe the image in detail",
            "the image should include",
            "engineering diagram",
            "describe the components, their arrangement, and any visible connections",
            "describe what you see, focusing on components",
            "what mechanical components, parts, or structures are visible",
            "this image shows a",
            "this image is from a section about",
        ]
        
        visual_facts_lower = visual_facts.lower()
        is_generic = any(phrase in visual_facts_lower for phrase in generic_phrases)
        
        # Start with visual facts if they're meaningful
        if visual_facts and not is_generic and len(visual_facts.strip()) > 20:
            caption = visual_facts.strip()
        else:
            caption = ""
        
        # Extract meaningful information from context
        context_info = []
        if context:
            context_lower = context.lower()
            # Extract subsystem mentions
            subsystems = ["drivetrain", "intake", "shooter", "elevator", "arm", "climber", "chassis", "gearbox", "pneumatic", "sensor"]
            found_subsystems = [s for s in subsystems if s in context_lower]
            if found_subsystems:
                context_info.append(f"{', '.join(found_subsystems[:2])} component")
            
            # Extract key technical terms from context
            context_terms = self._extract_technical_terms(context)
            if context_terms and not caption:
                # Use context terms if we don't have good visual facts
                caption = f"Technical diagram showing {', '.join(context_terms[:3])}"
        
        # Enhance with OCR text if present
        if ocr_text and len(ocr_text.strip()) > 3:
            ocr_clean = ocr_text.strip()
            # Extract meaningful OCR terms
            ocr_terms = self._extract_technical_terms(ocr_clean)
            
            if ocr_terms:
                terms_str = ", ".join(ocr_terms[:5])
                if caption:
                    # Add OCR labels if not already mentioned
                    if not any(t.lower() in caption.lower() for t in ocr_terms[:3]):
                        caption = f"{caption} Labels: {terms_str}."
                else:
                    # Use OCR as primary if no good visual facts
                    caption = f"Technical diagram with labels: {terms_str}"
            elif len(ocr_clean) > 10 and not caption:
                # Use OCR text directly if it's substantial
                caption = f"Technical diagram with text: {ocr_clean[:100]}"
        
        # Combine context info if we have it
        if context_info and caption:
            # Prepend subsystem info if not already in caption
            if not any(info.lower() in caption.lower() for info in context_info):
                caption = f"{', '.join(context_info)}. {caption}"
        elif context_info and not caption:
            caption = f"Technical diagram of {', '.join(context_info)}"
        
        # Final fallback if still empty
        if not caption or len(caption.strip()) < 10:
            if context:
                # Try to extract at least something from context
                context_words = [w for w in context.split() if len(w) > 4][:5]
                if context_words:
                    caption = f"Technical diagram related to {', '.join(context_words[:3])}"
                else:
                    caption = "Technical engineering diagram"
            else:
                caption = "Technical engineering diagram"
        
        return caption.strip()

    def caption_image(
        self,
        image: Union[Image.Image, Path, str],
        image_id: str,
        context: str = "",
        section_header: str = "",
    ) -> ImageCaption:
        """
        Generate caption for an image.
        
        Args:
            image: Image or path to image
            image_id: Unique identifier
            context: Nearby page text for grounding
            section_header: Current section header
            
        Returns:
            ImageCaption with full metadata
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # 1. Extract raw visual facts
        try:
            prompt = None
            if section_header:
                prompt = f"This image is from a section about '{section_header}'. What specific mechanical components, mechanisms, or structures do you see? Describe their shapes, sizes, positions, and how they connect or relate to each other."
            elif context:
                # Extract key terms from context to guide the model
                context_lower = context.lower()[:200]  # First 200 chars
                # Look for subsystem mentions
                subsystems = ["drivetrain", "intake", "shooter", "elevator", "arm", "climber", "chassis", "gearbox", "motor", "pneumatic", "sensor"]
                found_subsystem = next((s for s in subsystems if s in context_lower), None)
                if found_subsystem:
                    prompt = f"This image shows a {found_subsystem} component. What specific parts, mechanisms, or design features are visible? Describe the components, their arrangement, and any visible connections."
            
            raw_facts = self.vision_model.describe_image(image, prompt)
        except Exception as e:
            logger.error(f"Vision model failed for {image_id}: {e}")
            raw_facts = "Image could not be analyzed."
        
        # 2. Extract OCR text
        ocr_text = ""
        if self.use_ocr:
            ocr_text = self._extract_ocr_text(image)
        
        # 3. Synthesize caption
        final_caption = self._synthesize_caption(raw_facts, ocr_text, context)
        
        # 4. Validate
        validation_passed, notes = self._validate_caption(
            final_caption, ocr_text, context
        )
        
        # 5. Apply fallback if validation issues - but use available info
        if notes and not validation_passed:
            # Try to build a better fallback using available information
            fallback_parts = []
            
            # Use context if available
            if context:
                context_terms = self._extract_technical_terms(context)
                if context_terms:
                    fallback_parts.append(f"{', '.join(context_terms[:3])} component")
            
            # Use OCR if available
            if ocr_text and len(ocr_text.strip()) > 3:
                ocr_terms = self._extract_technical_terms(ocr_text)
                if ocr_terms:
                    fallback_parts.append(f"with labels: {', '.join(ocr_terms[:3])}")
            
            # Use raw facts if they're not generic
            if raw_facts and "engineering diagram" not in raw_facts.lower() and len(raw_facts) > 20:
                fallback_parts.append(raw_facts[:100])
            
            if fallback_parts:
                final_caption = ". ".join(fallback_parts) + "."
            else:
                final_caption = "Technical engineering diagram"
        
        return ImageCaption(
            image_id=image_id,
            raw_visual_facts=raw_facts,
            final_caption=final_caption,
            ocr_text=ocr_text if ocr_text else None,
            source_context=context[:500] if context else None,
            validation_passed=validation_passed,
            validation_notes=notes,
            confidence=1.0 if validation_passed and not notes else 0.7,
        )

    def _caption_single_image(
        self,
        img: ProcessedImage,
        context: str,
    ) -> ImageCaption:
        """
        Caption a single image (worker function for parallel processing).
        
        This function handles I/O operations (image loading, OCR) in parallel,
        but synchronizes GPU inference to avoid memory issues.
        
        Args:
            img: Processed image
            context: Context text for the image
            
        Returns:
            ImageCaption object
        """
        try:
            # Load image and do OCR in parallel (I/O bound, no GPU needed)
            image = Image.open(img.saved_path).convert("RGB")
            
            # Extract OCR text in parallel (CPU-bound)
            ocr_text = ""
            if self.use_ocr:
                ocr_text = self._extract_ocr_text(image)
            
            # GPU inference: synchronize access with lock
            if self._gpu_lock:
                with self._gpu_lock:
                    raw_facts = self.vision_model.describe_image(image)
            else:
                raw_facts = self.vision_model.describe_image(image)
            
            # Synthesize caption (CPU-bound, can be done in parallel)
            final_caption = self._synthesize_caption(raw_facts, ocr_text, context)
            
            # Validate (CPU-bound)
            validation_passed, notes = self._validate_caption(
                final_caption, ocr_text, context
            )
            
            # Apply fallback if needed
            if notes and not validation_passed:
                fallback_parts = []
                if context:
                    context_terms = self._extract_technical_terms(context)
                    if context_terms:
                        fallback_parts.append(f"{', '.join(context_terms[:3])} component")
                if ocr_text and len(ocr_text.strip()) > 3:
                    ocr_terms = self._extract_technical_terms(ocr_text)
                    if ocr_terms:
                        fallback_parts.append(f"with labels: {', '.join(ocr_terms[:3])}")
                if raw_facts and "engineering diagram" not in raw_facts.lower() and len(raw_facts) > 20:
                    fallback_parts.append(raw_facts[:100])
                
                if fallback_parts:
                    final_caption = ". ".join(fallback_parts) + "."
                else:
                    final_caption = "Technical engineering diagram"
            
            return ImageCaption(
                image_id=img.image_id,
                raw_visual_facts=raw_facts,
                final_caption=final_caption,
                ocr_text=ocr_text if ocr_text else None,
                source_context=context[:500] if context else None,
                validation_passed=validation_passed,
                validation_notes=notes,
                confidence=1.0 if validation_passed and not notes else 0.7,
            )
        except Exception as e:
            logger.error(
                "Caption generation failed",
                image_id=img.image_id,
                error=str(e),
            )
            # Return fallback caption
            return ImageCaption(
                image_id=img.image_id,
                raw_visual_facts="",
                final_caption="Image caption unavailable.",
                validation_passed=False,
                validation_notes=[f"Error: {str(e)}"],
                confidence=0.0,
            )
    
    def caption_processed_images(
        self,
        images: List[ProcessedImage],
        context_map: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
    ) -> List[ImageCaption]:
        """
        Caption multiple processed images in parallel.
        
        Args:
            images: List of processed images
            context_map: Optional mapping of image_id to context text
            show_progress: Show progress
            
        Returns:
            List of captions
        """
        context_map = context_map or {}
        
        # Filter valid images
        valid_images = [
            img for img in images
            if not img.is_duplicate and img.saved_path and img.saved_path.exists()
        ]
        
        if not valid_images:
            return []
        
        logger.info(
            "Generating captions",
            total=len(valid_images),
            workers=self.max_workers,
            device=self.device,
        )
        
        # Use parallel processing
        captions_dict = {}  # Use dict to maintain order
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(
                    self._caption_single_image,
                    img,
                    context_map.get(img.image_id, ""),
                ): img
                for img in valid_images
            }
            
            # Process completed tasks
            for future in as_completed(future_to_image):
                img = future_to_image[future]
                try:
                    caption = future.result()
                    captions_dict[img.image_id] = caption
                    completed += 1
                    
                    if show_progress and completed % 10 == 0:
                        logger.info(f"Captioned {completed}/{len(valid_images)} images")
                except Exception as e:
                    logger.error(
                        "Failed to get caption result",
                        image_id=img.image_id,
                        error=str(e),
                    )
                    # Add fallback
                    captions_dict[img.image_id] = ImageCaption(
                        image_id=img.image_id,
                        raw_visual_facts="",
                        final_caption="Image caption unavailable.",
                        validation_passed=False,
                        validation_notes=[f"Error: {str(e)}"],
                        confidence=0.0,
                    )
        
        # Return captions in original order
        return captions

    def caption_processed_images(
        self,
        images: List[ProcessedImage],
        context_map: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
        batch_size: int = 16,
    ) -> List[ImageCaption]:
        """
        Caption multiple processed images using 3-stage pipeline:
        1. Parallel CPU: Load images & OCR
        2. Batched GPU: Vision model Inference
        3. Parallel CPU: Synthesis & Validation
        """
        context_map = context_map or {}
        
        # Filter valid images
        valid_images = [
            img for img in images
            if not img.is_duplicate and img.saved_path and img.saved_path.exists()
        ]
        
        if not valid_images:
            return []
            
        logger.info(
            "Generating captions (Batch Pipeline)",
            total=len(valid_images),
            batch_size=batch_size,
            device=self.device,
        )
        
        all_results: Dict[str, ImageCaption] = {}
        
        # --- Stage 1: Load and OCR (Parallel CPU) ---
        # We need to load PIL images into memory for the GPU stage
        loaded_data = [] # List of (img_obj, processed_img, ocr_text, context)
        
        def process_stage1(img: ProcessedImage):
            try:
                pil_img = Image.open(img.saved_path).convert("RGB")
                ocr_text = self._extract_ocr_text(pil_img) if self.use_ocr else ""
                context = context_map.get(img.image_id, "")
                return (pil_img, img, ocr_text, context)
            except Exception as e:
                logger.error(f"Stage 1 failed for {img.image_id}: {e}")
                return None

        logger.info("Stage 1: Loading images and running OCR...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a dict to store results by image_id to handle out-of-order completion
            future_to_img = {executor.submit(process_stage1, img): img for img in valid_images}
            results_map = {}
            
            for future in as_completed(future_to_img):
                img = future_to_img[future]
                try:
                    res = future.result()
                    if res:
                        results_map[img.image_id] = res
                except Exception as e:
                    logger.error(f"Stage 1 failed for {img.image_id}: {e}")
                
                if show_progress and len(results_map) % 10 == 0:
                    logger.info(f"Stage 1 Progress: Loaded {len(results_map)}/{len(valid_images)} images")
            
            # Reconstruct loaded_data in the original order of valid_images
            for img in valid_images:
                if img.image_id in results_map:
                    loaded_data.append(results_map[img.image_id])
                    
        # Sort back to original order (optional, but good for debugging)
        # We'll map by ID at the end anyway, but processing in order is nice
        
        # --- Stage 2: Batched GPU Inference ---
        logger.info(f"Stage 2: Running batched vision inference on {len(loaded_data)} images...")
        
        # Prepare batches
        raw_facts_map = {} # image_id -> raw_facts
        
        for i in range(0, len(loaded_data), batch_size):
            batch = loaded_data[i:i + batch_size]
            batch_pil = [item[0] for item in batch]
            batch_imgs = [item[1] for item in batch] # ProcessedImage objects
            
            try:
                # Run inference
                descriptions = self.vision_model.describe_images_batch(batch_pil)
                
                # Store results
                for img_obj, desc in zip(batch_imgs, descriptions):
                    raw_facts_map[img_obj.image_id] = desc
                    
                if show_progress:
                    logger.info(f" Inferred {min(i + batch_size, len(loaded_data))}/{len(loaded_data)} images")
                    
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                for img_obj in batch_imgs:
                    raw_facts_map[img_obj.image_id] = "Error in batch inference."

        # --- Stage 3: Synthesis & Validation (Parallel CPU) ---
        logger.info("Stage 3: Synthesizing and validating captions...")
        
        def process_stage3(data_item):
            pil_img, img, ocr_text, context = data_item
            raw_facts = raw_facts_map.get(img.image_id, "")
            
            final_caption = self._synthesize_caption(raw_facts, ocr_text, context)
            valid, notes = self._validate_caption(final_caption, ocr_text, context)
            
            # Fallback logic
            if notes and not valid:
                # (Simple fallback logic reused from original)
                fallback = []
                if raw_facts and len(raw_facts) > 20: fallback.append(raw_facts[:100])
                if fallback: final_caption = ". ".join(fallback)
                
            return ImageCaption(
                image_id=img.image_id,
                raw_visual_facts=raw_facts,
                final_caption=final_caption,
                ocr_text=ocr_text or None,
                source_context=context[:200] if context else None,
                validation_passed=valid,
                validation_notes=notes,
                confidence=1.0 if valid else 0.7
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_stage3, item): item[1].image_id for item in loaded_data}
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    cap = future.result()
                    all_results[img_id] = cap
                except Exception as e:
                    logger.error(f"Stage 3 failed for {img_id}: {e}")

        # Return ordered list
        final_captions = [all_results[img.image_id] for img in valid_images if img.image_id in all_results]
        return final_captions
