"""
Query Processor - Integrates game piece mapping with RAG system
This module processes user queries and modify them with game piece context
"""

import os
import subprocess
import time
import requests
import numpy as np
import re
import chromadb
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from .game_piece_mapper import GamePieceMapper
from .image_embedder import ImageEmbedder
from ..utils.query_cache import QueryCache, ChunkCache
from ..utils.feedback_manager import FeedbackManager
from ..server.openrouter_client import OpenRouterClient
from ..server.config import get_config
from ..utils.citation_formatter import format_web_citation, format_youtube_citation

Config = get_config()

class SimplePostProcessor:
    """
    Lightweight post-processing filter for improving retrieval precision
    """
    
    def __init__(self, min_relevance_score: float = 0.3):
        self.min_relevance_score = min_relevance_score
        
        # FRC-specific keywords for relevance scoring
        self.high_value_keywords = [
            'motor', 'gear', 'ratio', 'wheel', 'sensor', 'encoder', 'gyro',
            'autonomous', 'teleop', 'programming', 'pid', 'control', 'feedback',
            'intake', 'shooter', 'drivetrain', 'elevator', 'arm', 'chassis',
            'swerve', 'tank', 'camera', 'vision', 'apriltag', 'pathfinding'
        ]
        
        self.medium_value_keywords = [
            'design', 'build', 'material', 'aluminum', 'weight', 'strength',
            'power', 'battery', 'pneumatic', 'mechanical', 'electrical'
        ]
    
    def calculate_relevance_score(self, query: str, document: str) -> float:
        """Calculate relevance score between query and document"""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Extract words
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        doc_words = set(re.findall(r'\b\w+\b', doc_lower))
        
        # Calculate keyword overlap
        overlap = len(query_words.intersection(doc_words))
        keyword_score = overlap / len(query_words) if query_words else 0
        
        # Calculate FRC-specific relevance
        high_value_matches = sum(1 for kw in self.high_value_keywords if kw in doc_lower)
        medium_value_matches = sum(1 for kw in self.medium_value_keywords if kw in doc_lower)
        
        # Weighted FRC score
        frc_score = (high_value_matches * 3 + medium_value_matches * 2) / 100
        
        # Document length penalty
        doc_length = len(document.split())
        length_penalty = 1.0
        if doc_length < 50:
            length_penalty = 0.7
        elif doc_length > 1000:
            length_penalty = 0.8
        
        # Combine scores
        relevance_score = (keyword_score * 0.5 + frc_score * 0.5) * length_penalty
        return min(relevance_score, 1.0)
    
    def filter_documents(self, query: str, documents: List[str], target_count: int = 10) -> List[str]:
        """
        Filter documents based on relevance score
        
        Args:
            query: Search query
            documents: List of document content strings
            target_count: Maximum number of documents to return
            
        Returns:
            Filtered list of documents
        """
        if not documents:
            return []
        
        # Score all documents
        scored_docs = []
        for doc in documents:
            score = self.calculate_relevance_score(query, doc)
            if score >= self.min_relevance_score:
                scored_docs.append((doc, score))
        
        # Sort by score (descending) and limit count
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        filtered_docs = [doc for doc, score in scored_docs[:target_count]]
        
        if scored_docs:
            avg_score = np.mean([score for _, score in scored_docs])
            print(f"Post-processing: {len(documents)} -> {len(filtered_docs)} documents (avg score: {avg_score:.3f})")
        
        return filtered_docs

class QueryProcessor:
    def __init__(self, chroma_path: str = "db", images_path: str = "data/images", 
                 enable_cache: bool = True, cache_config: Dict[str, Any] = None,
                 enable_post_processing: bool = True, min_relevance_score: float = 0.3):
        self.chroma_path = chroma_path
        self.images_path = images_path
        self.game_piece_mapper = GamePieceMapper()
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        
        # Initialize post-processing filter
        self.enable_post_processing = enable_post_processing
        if self.enable_post_processing:
            try:
                self.post_processor = SimplePostProcessor(min_relevance_score=min_relevance_score)
                print("Post-processing filter enabled")
            except Exception as e:
                print(f"Warning: Could not initialize post-processor: {e}")
                self.enable_post_processing = False
                self.post_processor = None
        else:
            self.post_processor = None
        
        # Initialize cache system
        self.enable_cache = enable_cache
        if self.enable_cache:
            cache_config = cache_config or {}
            self.query_cache = QueryCache(
                max_size=cache_config.get('max_size', 1000),
                similarity_threshold=cache_config.get('similarity_threshold', 0.92),
                ttl_seconds=cache_config.get('ttl_seconds', 3600),
                enable_semantic_cache=cache_config.get('enable_semantic_cache', True)
            )
            self.chunk_cache = ChunkCache(
                max_size=cache_config.get('chunk_cache_size', 500),
                ttl_seconds=cache_config.get('chunk_ttl_seconds', 1800)
            )
            print("✅ Query cache system enabled")
        else:
            self.query_cache = None
            self.chunk_cache = None
            print("⚠️  Query cache system disabled")
            
        # Initialize SigLIP model for image search
        try:
            self.img_model = ImageEmbedder()
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.image_collection = self.chroma_client.get_collection(name="image_embeddings")
            self.enable_image_search = True
            print("✅ Image search initialized with SigLIP model")
        except Exception as e:
            print(f"Warning: Could not initialize image search: {e}")
            self.enable_image_search = False
        
        # Start Ollama service (only if using local provider)
        if Config.MODEL_PROVIDER == 'local':
            self._ensure_ollama_running()
        
        # Initialize model client
        self._init_model_client()
        
        # Initialize database
        self.db = None
        self._init_database()
        
        # Initialize feedback manager
        self.feedback_manager = FeedbackManager(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'))
        
        # Enhanced prompt template
        self.prompt_template = """
You are an expert FRC (FIRST Robotics Competition) assistant. Answer the question based on the following context and game piece information.

IMPORTANT:
- If "EXTERNAL SOURCES" are provided in the context, prioritize them for recent events, game names, and current season information (e.g., 2025, 2026).
- Technical documents might be from previous years. Always check the dates or source titles.
- If the answer is found in the external sources, cite them using the format [Source Name].

{conversation_history}

CONTEXT FROM TECHNICAL DOCUMENTS AND EXTERNAL SOURCES:
{context}

GAME PIECE INFORMATION:
{game_piece_context}

RELEVANT EXAMPLES FROM PAST FEEDBACK:
{feedback_examples}

---

Question: {question}

Instructions:
1. Use the technical documentation context to provide specific, actionable advice
2. If there is previous conversation history, consider it when answering (e.g., if the user asks "tell me more" or "what about that", refer to the previous context)
3. When discussing game pieces, acknowledge user terminology
3. Include relevant dimensions, specifications, and technical details from the context
4. If images are mentioned in the context, reference them in your answer
5. Connect the technical information with practical implementation advice
6. If the context doesn't provide complete information, use your FRC knowledge to fill gaps while clearly indicating what comes from the documents vs. general knowledge
7. If a user defined a game piece, say how it might be similar to dealing with a known piece
8. When answering queries about designs, give pros and cons based on the context, along with steps to implement with CAD screenshots or build tips
9. When referencing anything, make sure to attach the image if one is available
10. If an EXTERNAL SOURCES section is provided, treat those links/snippets as the highest-priority evidence: cite them explicitly and prefer their facts over older context. If web sources disagree, call it out and favor the most recent-looking source.
"""

    def _build_citation_section(self, web_citations, youtube_citations) -> str:
        if not web_citations and not youtube_citations:
            return ""
        lines = []
        for c in web_citations:
            lines.append(f"• {c}")
        for c in youtube_citations:
            lines.append(f"• {c}")
        return "\nEXTERNAL SOURCES:\n" + "\n".join(lines)

    def _ensure_ollama_running(self):
        """Ensure Ollama service is running and model is available"""
        print("🤖 Checking Ollama service...")
        
        # Check if Ollama is already running
        if self._is_ollama_running():
            print("✅ Ollama service is already running")
            self._ensure_model_available()
            return
        
        # Try to start Ollama
        print("🚀 Starting Ollama service...")
        try:
            # Start Ollama in the background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                if self._is_ollama_running():
                    print("✅ Ollama service started successfully")
                    self._ensure_model_available()
                    return
            
            print("⚠️ Ollama service took too long to start")
            
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama from https://ollama.ai/")
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _ensure_model_available(self):
        """Ensure the required model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if any('mistral' in name for name in model_names):
                    print("✅ Mistral model is available")
                    return
                
                # Model not found, try to pull it
                print("📥 Mistral model not found. Attempting to download...")
                try:
                    subprocess.run(['ollama', 'pull', 'mistral'], 
                                 check=True, 
                                 timeout=300,  # 5 minutes timeout
                                 capture_output=True)
                    print("✅ Mistral model downloaded successfully")
                except subprocess.TimeoutExpired:
                    print("⚠️ Model download timed out. You may need to run 'ollama pull mistral' manually")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to download model: {e}")
                except Exception as e:
                    print(f"❌ Error downloading model: {e}")
        except Exception as e:
            print(f"❌ Error checking models: {e}")

    def _init_database(self):
        """Initialize the ChromaDB connection"""
        if not os.path.exists(self.chroma_path):
            print(f"Database not found at {self.chroma_path}")
            return False
        
        try:
            self.db = Chroma(persist_directory=self.chroma_path, 
                           embedding_function=self.embedding_function)
            print(f"✅ Database initialized successfully from {self.chroma_path}")
            return True
        except Exception as e:
            print(f"Error initializing database: {e}")
            return False

    def _init_model_client(self):
        """Initialize the appropriate model client based on configuration"""
        provider = Config.MODEL_PROVIDER
        if provider == 'chute':
            provider = 'openrouter'

        if provider == 'openrouter':
            try:
                self.openrouter_client = OpenRouterClient()
                print(f"✅ OpenRouter client initialized for model provider: {provider}")
            except Exception as e:
                print(f"❌ Failed to initialize OpenRouter client: {e}")
                self.openrouter_client = None
        else:
            self.openrouter_client = None
            print(f"✅ Using local Ollama for model provider: {provider}")

    def _generate_response(self, prompt: str, show_reasoning: bool = None) -> str:
        """Generate response using the configured model provider"""
        provider = Config.MODEL_PROVIDER
        if provider == 'chute':
            provider = 'openrouter'

        if provider == 'openrouter' and getattr(self, 'openrouter_client', None):
            try:
                return self.openrouter_client.chat_completion(prompt, show_reasoning=show_reasoning)
            except Exception as e:
                print(f"❌ OpenRouter request failed: {e}")
                # Fallback to basic response
                return "I encountered an issue generating the response. Please try again."
        else:
            # Use Ollama
            try:
                model = Ollama(model="gpt-oss:20b")
                return model.invoke(prompt)
            except Exception as e:
                print(f"❌ Ollama request failed: {e}")
                return "I encountered an issue generating the response. Please try again."

    def _generate_response_stream(self, prompt: str, show_reasoning: bool = None, 
                                    custom_api_key: str = None, custom_model: str = None, system_prompt: str = None):
        """Generate streaming response using the configured model provider"""
        provider = Config.MODEL_PROVIDER
        if provider == 'chute':
            provider = 'openrouter'

        if provider == 'openrouter' and getattr(self, 'openrouter_client', None):
            try:
                for chunk in self.openrouter_client.chat_completion_stream(
                    prompt, 
                    show_reasoning=show_reasoning,
                    custom_api_key=custom_api_key,
                    custom_model=custom_model,
                    system_prompt=system_prompt
                ):
                    yield chunk
            except Exception as e:
                print(f"❌ OpenRouter streaming failed: {e}")
                yield "I encountered an issue generating the response. Please try again."
        else:
            # Use Ollama streaming
            try:
                model = Ollama(model="gpt-oss:20b")
                for chunk in model.stream(prompt):
                    yield chunk
            except Exception as e:
                print(f"❌ Ollama streaming failed: {e}")
                yield "I encountered an issue generating the response. Please try again."

    def process_query(self, query: str, k: int = 10, enable_filtering: bool = None, target_docs: int = 8, 
                     conversation_history: List[Dict] = None, enable_web_search: bool = False, enable_youtube_search: bool = False) -> Dict[str, Any]:
        """
        Process a user query with game piece enhancement, caching, and optional post-processing
        Returns a comprehensive response with context and metadata
        
        Args:
            query: The user's query
            k: Number of documents to retrieve
            enable_filtering: Whether to enable post-processing filtering
            target_docs: Target number of documents after filtering
            conversation_history: List of previous messages in format [{"role": "user|assistant", "content": "..."}]
        """
        if not self.db:
            return {"error": "Database not initialized"}
        
        # Determine if we should use filtering - only if explicitly enabled
        use_filtering = enable_filtering if enable_filtering is not None else False
        
        # If using filtering, retrieve more documents initially
        initial_k = k * 2 if use_filtering and self.post_processor else k
        
        # Check cache first (if enabled)
        if self.enable_cache and self.query_cache:
            # Generate query embedding for semantic cache lookup
            query_embedding = None
            try:
                query_embedding = np.array(self.embedding_function.embed_query(query))
            except Exception as e:
                print(f"Warning: Could not generate query embedding for cache: {e}")
            
            # Try to get cached response (use original k for cache key)
            cached_response = self.query_cache.get(query, k, query_embedding)
            if cached_response is not None:
                print(f"Cache hit ({cached_response.get('_cache_type', 'unknown')})")
                return cached_response
        
        # Cache miss - process query normally
        print("Processing new query (cache miss)")
        
        # Step 1: Analyze query for game pieces
        enhanced_query, matched_pieces = self.game_piece_mapper.enhance_query(query)
        
        # Ensure enhanced_query is a string
        if not isinstance(enhanced_query, str):
            enhanced_query = str(enhanced_query) if enhanced_query else query
        
        # Fall back to original query if enhanced query is empty
        if not enhanced_query.strip():
            enhanced_query = query
        
        # Step 2: Search database with enhanced query (using initial_k)
        try:
            # Generate embedding for the enhanced query
            enhanced_embedding = None
            if self.enable_cache and self.chunk_cache:
                try:
                    enhanced_embedding = np.array(self.embedding_function.embed_query(enhanced_query))
                    
                    # Try to get cached chunks (use initial_k for caching)
                    cached_chunks = self.chunk_cache.get(enhanced_embedding, initial_k)
                    if cached_chunks is not None:
                        print(f"Chunk cache hit")
                        results = cached_chunks
                    else:
                        # Perform similarity search
                        results = self.db.similarity_search(enhanced_query, k=initial_k)
                        # Cache the chunks
                        self.chunk_cache.set(enhanced_embedding, initial_k, results)
                except Exception as e:
                    print(f"Warning: Chunk cache error: {e}")
                    results = self.db.similarity_search(enhanced_query, k=initial_k)
            else:
                results = self.db.similarity_search(enhanced_query, k=initial_k)
        except Exception as e:
            return {"error": f"Search failed: {e}"}
        
        if not results:
            response = {
                "response": "I couldn't find relevant information in the database for your query.",
                "matched_pieces": matched_pieces,
                "enhanced_query": enhanced_query,
                "original_query": query,
                "context_sources": 0,
                "related_images": [],
                "game_piece_context": ""
            }
            
            # Cache the response (even if no results)
            if self.enable_cache and self.query_cache:
                try:
                    query_embedding = np.array(self.embedding_function.embed_query(query))
                    self.query_cache.set(query, response, k, query_embedding)
                except Exception as e:
                    print(f"Warning: Could not cache response: {e}")
            
            return response
        
        # Step 3: Collect context parts and apply post-processing filtering
        context_parts = [doc.page_content for doc in results]

        # --- Web and YouTube Search Integration ---
        web_results, youtube_results = [], []
        web_citations, youtube_citations = [], []
        print(f"[DEBUG] enable_web_search flag: {enable_web_search}")
        if enable_web_search:
            print(f"[DEBUG] Attempting SerpAPI web search for query: '{query}'")
            try:
                from ..utils.web_and_youtube_search import web_search
                web_results = web_search(query, num_results=3)
                print(f"[DEBUG] SerpAPI web_search results: {web_results}")
                web_citations = [format_web_citation(r) for r in web_results]
            except Exception as e:
                print(f"[ERROR] Web search error: {e}")

        # Merge citations into context
        citation_section = ""
        if web_citations or youtube_citations:
            citation_section = self._build_citation_section(web_citations, youtube_citations)

        # Apply post-processing filter if enabled
        if use_filtering and self.post_processor:
            filtered_context_parts = self.post_processor.filter_documents(
                query=query,
                documents=context_parts,
                target_count=target_docs
            )
            # If filtering removed everything, fall back to original (take top k)
            if not filtered_context_parts:
                print("Warning: All documents filtered out, using original top k results")
                filtered_context_parts = context_parts[:k]
            # Update context_parts and results to match filtered content
            context_parts = filtered_context_parts
            filtered_results = []
            for filtered_content in filtered_context_parts:
                for doc in results:
                    if doc.page_content == filtered_content:
                        filtered_results.append(doc)
                        break
            results = filtered_results
        else:
            context_parts = context_parts[:k]
            results = results[:k]

        # Step 4: Collect related images from filtered results
        related_images = []
        for doc in results:
            images = self._collect_images_from_result(doc)
            related_images.extend(images)

        # --- Image Search ---
        if self.enable_image_search:
            print("Searching for images using embeddings...")
            image_results = self.search_images(query, k=4)
            related_images.extend(image_results)

        # Remove duplicate images based on filename
        seen_filenames = set()
        unique_images = []
        for img in related_images:
            if img['filename'] not in seen_filenames:
                # Calculate relevance score for image
                relevance = None
                # If image has OCR/text, use that for relevance
                if 'ocr_text' in img and img['ocr_text']:
                    relevance = self.post_processor.calculate_relevance_score(query, img['ocr_text']) if self.post_processor else None
                elif 'formatted_context' in img and img['formatted_context']:
                    relevance = self.post_processor.calculate_relevance_score(query, img['formatted_context']) if self.post_processor else None
                img['relevance'] = relevance
                unique_images.append(img)
                seen_filenames.add(img['filename'])

        # Step 5: Generate game piece context
        game_piece_context = ""
        if matched_pieces:
            game_piece_context = self.game_piece_mapper.get_context_for_pieces(matched_pieces)

        # Step 6: Generate AI response using filtered context
        context_text = "\n\n---\n\n".join(context_parts)
        if citation_section:
            context_text += citation_section

        # Build conversation history text
        history_text = ""
        if conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                history_text += f"{role}: {content}\n"
            history_text += "\n---\n"

        # Get feedback examples
        feedback_examples = ""
        try:
            good_examples = self.feedback_manager.get_good_examples()
            if good_examples:
                feedback_text = []
                for ex in good_examples[:3]:
                    feedback_text.append(f"Q: {ex['query']}\nA: {ex['response']}")
                feedback_examples = "\n\n".join(feedback_text)
        except Exception as e:
            print(f"Warning: Could not get feedback examples: {e}")

        try:
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(
                context=context_text,
                game_piece_context=game_piece_context,
                feedback_examples=feedback_examples,
                conversation_history=history_text,
                question=query
            )
            
            response_text = self._generate_response(prompt)
            
        except Exception as e:
            # Fallback response if AI generation fails
            response_text = f"Based on the technical documentation, here's what I found:\n\n{context_text[:1000]}..."
            if game_piece_context:
                response_text += f"\n\nGame Piece Information:\n{game_piece_context}"

        # Prepare structured external sources for frontend (consistent with prepare_query_metadata)
        external_sources = []
        for r in web_results:
            external_sources.append({
                "type": "web",
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "source": "Web"
            })
        for r in youtube_results: # Note: youtube_results was populated earlier if enabled
             # Wait, youtube_results is populated in the web search block above?
             # Let's check lines 460-480
             pass

        response = {
            "response": response_text,
            "matched_pieces": matched_pieces,
            "enhanced_query": enhanced_query,
            "original_query": query,
            "context_sources": len(context_parts),  # Use filtered count
            "related_images": unique_images,
            "game_piece_context": game_piece_context,
            "external_sources_section": citation_section if citation_section else "",
            "external_sources": external_sources
        }
        
        # Add post-processing information if applied
        if use_filtering and self.post_processor:
            response["post_processing_applied"] = True
            response["initial_doc_count"] = initial_k
            response["filtered_doc_count"] = len(context_parts)
        
        # Cache the response (with original k as cache key)
        if self.enable_cache and self.query_cache:
            try:
                query_embedding = np.array(self.embedding_function.embed_query(query))
                self.query_cache.set(query, response, k, query_embedding)
                print("Response cached")
            except Exception as e:
                print(f"Warning: Could not cache response: {e}")
        
        return response

    def _collect_images_from_result(self, doc) -> List[Dict[str, Any]]:
        """Collect image information from a document result"""
        images_info = []
        
        try:
            # Check if this is an image context document (preferred)
            if doc.metadata.get('type') in ['image_context', 'enhanced_image_context']:
                image_file = doc.metadata.get('image_file')
                image_path = doc.metadata.get('image_path')
                if image_file and image_path:
                    context_excerpt = doc.metadata.get('page_context_excerpt', '')
                    if isinstance(context_excerpt, (list, tuple)):
                        context_excerpt = str(context_excerpt)
                    
                    images_info.append({
                        'filename': image_file,
                        'file_path': image_path,
                        'page': doc.metadata.get('page'),
                        'ocr_text': doc.metadata.get('image_text', ''),
                        'formatted_context': doc.metadata.get('formatted_context', ''),
                        'context_summary': context_excerpt[:200] if context_excerpt else ''
                    })
            
            # Check if this is an image document (fallback)
            elif doc.metadata.get('type') in ['image_text', 'enhanced_image_text', 'enhanced_image_info']:
                image_file = doc.metadata.get('image_file')
                image_path = doc.metadata.get('image_path')
                if image_file and image_path:
                    try:
                        ocr_content = doc.page_content
                        if isinstance(ocr_content, str):
                            ocr_text = ocr_content.replace('Image content: ', '').split('\n\nContext:')[0]
                        else:
                            ocr_text = str(ocr_content)
                    except Exception as e:
                        print(f"Warning: Error processing OCR content for {image_file}: {e}")
                        ocr_text = str(doc.page_content) if doc.page_content else ''
                    
                    images_info.append({
                        'filename': image_file,
                        'file_path': image_path,
                        'page': doc.metadata.get('page'),
                        'ocr_text': ocr_text,
                        'formatted_context': '',
                        'context_summary': ocr_text[:200] if ocr_text else ''
                    })
            
            # Check for text documents with associated images
            elif doc.metadata.get('type') in ['text_with_images', 'enhanced_text_with_images']:
                import json
                image_filenames_str = doc.metadata.get('image_filenames', '[]')
                try:
                    if isinstance(image_filenames_str, str):
                        image_filenames = json.loads(image_filenames_str)
                    elif isinstance(image_filenames_str, (list, tuple)):
                        image_filenames = list(image_filenames_str)
                    else:
                        image_filenames = []
                    
                    source_path = doc.metadata.get('source', '')
                    pdf_name = os.path.splitext(os.path.basename(source_path))[0] if source_path else ''
                    
                    for filename in image_filenames:
                        if pdf_name:
                            file_path = os.path.join(self.images_path, pdf_name, filename)
                        else:
                            file_path = os.path.join(self.images_path, filename)
                        
                        images_info.append({
                            'filename': filename,
                            'file_path': file_path,
                            'page': doc.metadata.get('page'),
                            'ocr_text': '',
                            'formatted_context': '',
                            'context_summary': 'Associated with relevant text content'
                        })
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: Error processing image filenames: {e}")
                    pass
        
        except Exception as e:
            print(f"Warning: Error collecting images from document: {e}")
            print(f"Document type: {doc.metadata.get('type', 'unknown')}")
            
        return images_info

    def get_game_piece_suggestions(self, partial_query: str) -> List[Dict[str, str]]:
        suggestions = []
        partial_lower = partial_query.lower()
        
        for piece_id, piece_data in self.game_piece_mapper.game_pieces.items():
            # Check official name
            if partial_lower in piece_data["official_name"].lower():
                suggestions.append({
                    "type": "official_name",
                    "text": piece_data["official_name"],
                    "season": piece_data["season"],
                    "piece_id": piece_id
                })
            
            # Check aliases
            for alias in piece_data["aliases"]:
                if partial_lower in alias.lower():
                    suggestions.append({
                        "type": "alias",
                        "text": alias,
                        "official_name": piece_data["official_name"],
                        "season": piece_data["season"],
                        "piece_id": piece_id
                    })
            
            # Check descriptions
            if partial_lower in piece_data["description"].lower():
                suggestions.append({
                    "type": "description_match",
                    "text": piece_data["official_name"],
                    "season": piece_data["season"],
                    "piece_id": piece_id,
                    "match_context": piece_data["description"][:100] + "..."
                })
        
        # Remove duplicates and limit results
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = (suggestion["text"], suggestion["season"])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:8]  # Limit to 8 suggestions

    def search_by_season(self, season: str, query: str = "", k: int = 10) -> Dict[str, Any]:
        """Search for content from a specific season"""
        if not self.db:
            return {"error": "Database not initialized"}
        
        # Create season-specific filter
        season_filter = {"season": season}
        
        try:
            if query:
                results = self.db.similarity_search(query, k=k, filter=season_filter)
            else:
                # Get all documents from this season
                results = self.db.get(where=season_filter, limit=k)
                results = [type('obj', (object,), {'page_content': doc, 'metadata': meta}) 
                          for doc, meta in zip(results['documents'], results['metadatas'])]
        except Exception as e:
            return {"error": f"Season search failed: {e}"}
        
        return self.process_query(query or f"Show me information from {season} season", k)

    def get_available_seasons(self) -> List[str]:
        """Get list of available seasons in the database"""
        return list(self.game_piece_mapper.seasons.keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary containing cache statistics including hit rates
        """
        if not self.enable_cache or not self.query_cache:
            return {
                "enabled": False,
                "message": "Cache is disabled"
            }
        
        query_stats = self.query_cache.get_stats()
        chunk_stats = self.chunk_cache.get_stats() if self.chunk_cache else {}
        
        return {
            "enabled": True,
            "query_cache": query_stats,
            "chunk_cache": chunk_stats,
            "total_memory_saved": {
                "description": "Estimated queries saved from reprocessing",
                "count": query_stats['total_hits']
            }
        }
    
    def clear_cache(self):
        """Clear all cache entries"""
        if self.enable_cache:
            if self.query_cache:
                self.query_cache.clear()
            if self.chunk_cache:
                self.chunk_cache.clear()
            print("✅ Cache cleared")
        else:
            print("⚠️  Cache is disabled")
    
    def reset_cache_stats(self):
        """Reset cache statistics"""
        if self.enable_cache:
            if self.query_cache:
                self.query_cache.reset_stats()
            print("✅ Cache stats reset")
        else:
            print("⚠️  Cache is disabled")
    
    def remove_expired_cache_entries(self):
        """Remove expired cache entries"""
        if self.enable_cache and self.query_cache:
            self.query_cache.remove_expired()
            print("✅ Expired cache entries removed")
        else:
            print("⚠️  Cache is disabled")    
    def prepare_query_metadata(self, query: str, k: int = 10, enable_filtering: bool = None, 
                              conversation_history: List[Dict] = None,
                              enable_web_search: bool = False,
                              enable_youtube_search: bool = False,
                              custom_serpapi_key: str = None,
                              custom_youtube_key: str = None) -> Dict[str, Any]:
        print(f"[DEBUG] prepare_query_metadata web_search={enable_web_search} youtube_search={enable_youtube_search}")
        """
        Prepare metadata for a query (images, matched pieces, etc.) without generating the response.
        This is used for streaming to send metadata first.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            enable_filtering: Whether to enable post-processing filtering (None = use default)
            conversation_history: List of previous messages in format [{"role": "user|assistant", "content": "..."}]
        """
        # Step 1: Enhance query with game piece context
        enhanced_query, matched_pieces = self.game_piece_mapper.enhance_query(query)
        
        # Ensure enhanced_query is a string
        if not isinstance(enhanced_query, str):
            enhanced_query = str(enhanced_query) if enhanced_query else query
        
        if not enhanced_query.strip():
            enhanced_query = query
        
        # Step 2: Search database with enhanced query
        try:
            enhanced_embedding = None
            if self.enable_cache and self.chunk_cache:
                try:
                    enhanced_embedding = np.array(self.embedding_function.embed_query(enhanced_query))
                    cached_chunks = self.chunk_cache.get(enhanced_embedding, k)
                    if cached_chunks is not None:
                        results = cached_chunks
                    else:
                        # Use k*2 to get more results for filtering
                        search_k = k * 2 if enable_filtering else k
                        results = self.db.similarity_search(enhanced_query, k=search_k)
                        self.chunk_cache.set(enhanced_embedding, k, results)
                except Exception as e:
                    search_k = k * 2 if enable_filtering else k
                    results = self.db.similarity_search(enhanced_query, k=search_k)
            else:
                search_k = k * 2 if enable_filtering else k
                results = self.db.similarity_search(enhanced_query, k=search_k)
                
            # Apply post-processing filtering if enabled
            if enable_filtering and hasattr(self, 'post_processor') and self.post_processor:
                filtered_results = self.post_processor.filter_documents(results, enhanced_query)
                
                if len(filtered_results) >= k:
                    results = filtered_results[:k]
                    print(f"Post-processing: {len(results)} -> {len(filtered_results[:k])} documents (avg score: {np.mean([self.post_processor._calculate_relevance_score(doc.page_content, enhanced_query) for doc in filtered_results[:k]]):.3f})")
                else:
                    print(f"Warning: Filtering returned {len(filtered_results)} docs, keeping original top {k}")
                    results = results[:k]
            else:
                results = results[:k]
                
        except Exception as e:
            return {"error": f"Search failed: {e}"}
        
        if not results:
            return {
                "matched_pieces": matched_pieces,
                "enhanced_query": enhanced_query,
                "original_query": query,
                "context_sources": 0,
                "images": [],
                "game_piece_context": "",
                "context_parts": [],
                "conversation_history": conversation_history or []
            }
        
        # Step 3: Collect related images and prepare context
        related_images = []
        context_parts = []
        
        for doc in results:
            context_parts.append(doc.page_content)
            images = self._collect_images_from_result(doc)
            related_images.extend(images)
        
        # Remove duplicate images and compute relevance
        seen_filenames = set()
        unique_images = []
        for img in related_images:
            if img['filename'] not in seen_filenames:
                relevance = None
                if self.post_processor:
                    text_basis = img.get('ocr_text') or img.get('formatted_context') or img.get('context_summary')
                    if text_basis:
                        relevance = self.post_processor.calculate_relevance_score(query, text_basis)
                img['relevance'] = relevance
                unique_images.append(img)
                seen_filenames.add(img['filename'])

        # Add embedding-based image search results (SigLIP) for metadata flow
        if self.enable_image_search:
            try:
                embed_images = self.search_images(query, k=4)
                for img in embed_images:
                    relevance = None
                    if self.post_processor:
                        text_basis = img.get('ocr_text') or img.get('formatted_context') or img.get('context_summary')
                        if text_basis:
                            relevance = self.post_processor.calculate_relevance_score(query, text_basis)
                    img['relevance'] = relevance
                    if img['filename'] not in seen_filenames:
                        unique_images.append(img)
                        seen_filenames.add(img['filename'])
            except Exception as e:
                print(f"Warning: embedding image search failed: {e}")
        
        # Process images for web display
        from ..server.config import get_config
        Config = get_config()
        
        web_images = []
        for img in unique_images:
            img_exists = os.path.exists(img['file_path'])
            web_path = img['file_path'].replace(Config.IMAGES_PATH + '/', '')
            
            web_images.append({
                'filename': img['filename'],
                'file_path': img['file_path'],
                'web_path': web_path,
                'page': img.get('page'),
                'exists': img_exists,
                'ocr_text': img.get('ocr_text', ''),
                'formatted_context': img.get('formatted_context', ''),
                'context_summary': img.get('context_summary', img.get('ocr_text', '')[:200] + ('...' if len(img.get('ocr_text', '')) > 200 else '')),
                'relevance': img.get('relevance')
            })

        # Web/YouTube search for metadata
        web_results = []
        yt_results = []
        web_citations, youtube_citations = [], []
        citation_section = ""
        
        if enable_web_search:
            try:
                from ..utils.web_and_youtube_search import web_search
                web_results = web_search(query, num_results=3, api_key=custom_serpapi_key)
                print(f"[DEBUG] Metadata web_search results: {len(web_results)}")
                web_citations = [format_web_citation(r) for r in web_results]
            except Exception as e:
                print(f"[ERROR] Web search error (metadata): {e}")
        
        if enable_youtube_search:
            try:
                from ..utils.web_and_youtube_search import youtube_search
                yt_results = youtube_search(query, num_results=2, api_key=custom_youtube_key)
                print(f"[DEBUG] Metadata youtube_search results: {len(yt_results)}")
                youtube_citations = [format_youtube_citation(r) for r in yt_results]
            except Exception as e:
                print(f"[ERROR] YouTube search error (metadata): {e}")

        if web_citations or youtube_citations:
            citation_section = self._build_citation_section(web_citations, youtube_citations)
        
        # Prepare structured external sources for frontend
        external_sources = []
        for r in web_results:
            external_sources.append({
                "type": "web",
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "source": "Web"
            })
        for r in yt_results:
            external_sources.append({
                "type": "youtube",
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("description"),
                "channel": r.get("channel"),
                "source": "YouTube"
            })

        # Generate game piece context
        game_piece_context = ""
        if matched_pieces:
            game_piece_context = self.game_piece_mapper.get_context_for_pieces(matched_pieces)
        
        return {
            "matched_pieces": matched_pieces,
            "enhanced_query": enhanced_query,
            "original_query": query,
            "context_sources": len(results),
            "images": web_images,
            "images_count": len(web_images),
            "game_piece_context": game_piece_context,
            "context_parts": context_parts,
            "conversation_history": conversation_history or [],
            "external_sources_section": citation_section,
            "external_sources": external_sources
        }
    
    def stream_query_response(self, query: str, metadata: Dict[str, Any], show_reasoning: bool = None,
                               custom_api_key: str = None, custom_model: str = None, system_prompt: str = None,
                               enable_web_search: bool = False, enable_youtube_search: bool = False):
        """
        Stream the LLM response for a query.
        Yields chunks of text as they're generated.
        
        Args:
            query: The user's query
            metadata: Query metadata from prepare_query_metadata
            show_reasoning: Whether to include reasoning content
            custom_api_key: Optional custom Chutes API key
            custom_model: Optional custom model to use
            system_prompt: Optional system prompt override
        """
        context_text = "\n\n---\n\n".join(metadata['context_parts'])
        citation_section = metadata.get('external_sources_section', '')
        if citation_section:
            context_text += citation_section
        game_piece_context = metadata.get('game_piece_context', '')
        conversation_history = metadata.get('conversation_history', [])
        
        # Build conversation history text
        history_text = ""
        if conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            # Include last 6 messages (3 exchanges) to keep context manageable
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')
                # Truncate very long messages to save tokens
                if len(content) > 300:
                    content = content[:300] + "..."
                history_text += f"{role}: {content}\n"
            history_text += "\n---\n"
        
        # Get feedback examples
        feedback_examples = ""
        try:
            good_examples = self.feedback_manager.get_good_examples()
            if good_examples:
                # Simple filtering: just take the last 3
                feedback_text = []
                for ex in good_examples[:3]:
                    feedback_text.append(f"Q: {ex['query']}\nA: {ex['response']}")
                feedback_examples = "\n\n".join(feedback_text)
        except Exception as e:
            print(f"Warning: Could not get feedback examples: {e}")

        try:
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(
                context=context_text,
                game_piece_context=game_piece_context,
                feedback_examples=feedback_examples,
                conversation_history=history_text,
                question=query
            )
            
            # Stream the response using the configured provider
            for chunk in self._generate_response_stream(
                prompt, 
                show_reasoning=show_reasoning,
                custom_api_key=custom_api_key,
                custom_model=custom_model,
                system_prompt=system_prompt
            ):
                yield chunk
                
        except Exception as e:
            # Fallback response if AI generation fails
            fallback = f"Based on the technical documentation, here's what I found:\n\n{context_text[:1000]}..."
            if game_piece_context:
                fallback += f"\n\nGame Piece Information:\n{game_piece_context}"
            
            # Yield fallback in chunks for consistent streaming experience
            chunk_size = 50
            for i in range(0, len(fallback), chunk_size):
                yield fallback[i:i+chunk_size]
                time.sleep(0.01)  # Small delay for visual effect
    
    def search_images(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for images using SigLIP embeddings
        """
        if not self.enable_image_search:
            return []
            
        try:
            # Encode query text
            query_emb = self.img_model.embed_text(query)[0]
            
            # Search
            results = self.image_collection.query(
                query_embeddings=[query_emb],
                n_results=k
            )
            
            # Format results
            images = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    # Reconstruct image object structure expected by frontend
                    img_entry = {
                        "filename": meta.get('filename', ''),
                        "file_path": meta.get('file_path', ''), # This should be the web path
                        "web_path": meta.get('file_path', ''), # Ensure web_path is set
                        "page": meta.get('page', 'N/A'),
                        "context_summary": results['documents'][0][i] if results['documents'] else '', # Use stored text as context
                        "score": results['distances'][0][i] if 'distances' in results else 0,
                        "exists": True # Flag for frontend
                    }
                    images.append(img_entry)
            return images
            
        except Exception as e:
            print(f"Error searching images: {e}")
            return []
