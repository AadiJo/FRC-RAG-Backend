"""
Query Processor - Integrates game piece mapping with RAG system
This module processes user queries and modify them with game piece context
"""

import os
import subprocess
import time
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from .game_piece_mapper import GamePieceMapper
from ..utils.query_cache import QueryCache, ChunkCache

class QueryProcessor:
    def __init__(self, chroma_path: str = "db", images_path: str = "data/images", 
                 enable_cache: bool = True, cache_config: Dict[str, Any] = None):
        self.chroma_path = chroma_path
        self.images_path = images_path
        self.game_piece_mapper = GamePieceMapper()
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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
        
        # Start Ollama service
        self._ensure_ollama_running()
        
        # Initialize database
        self.db = None
        self._init_database()
        
        # Enhanced prompt template
        self.prompt_template = """
You are an expert FRC (FIRST Robotics Competition) assistant. Answer the question based on the following context and game piece information:

CONTEXT FROM TECHNICAL DOCUMENTS:
{context}

GAME PIECE INFORMATION:
{game_piece_context}

---

Question: {question}

Instructions:
1. Use the technical documentation context to provide specific, actionable advice
2. When discussing game pieces, acknowledge user terminology
3. Include relevant dimensions, specifications, and technical details from the context
4. If images are mentioned in the context, reference them in your answer
5. Connect the technical information with practical implementation advice
6. If the context doesn't provide complete information, use your FRC knowledge to fill gaps while clearly indicating what comes from the documents vs. general knowledge
7. If a user defined a game piece, say how it might be similar to dealing with a known piece
8. When answering queries about designs, give pros and cons based on the context, along with steps to implement with CAD screenshots or build tips
9. When referencing anything, make sure to attach the image if one is available
"""

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

    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query with game piece enhancement and caching
        Returns a comprehensive response with context and metadata
        """
        if not self.db:
            return {"error": "Database not initialized"}
        
        # Check cache first (if enabled)
        if self.enable_cache and self.query_cache:
            # Generate query embedding for semantic cache lookup
            query_embedding = None
            try:
                query_embedding = np.array(self.embedding_function.embed_query(query))
            except Exception as e:
                print(f"Warning: Could not generate query embedding for cache: {e}")
            
            # Try to get cached response
            cached_response = self.query_cache.get(query, k, query_embedding)
            if cached_response is not None:
                print(f"✅ Cache hit ({cached_response.get('_cache_type', 'unknown')})")
                return cached_response
        
        # Cache miss - process query normally
        print("🔍 Processing new query (cache miss)")
        
        # Step 1: Analyze query for game pieces
        matched_pieces, enhanced_query = self.game_piece_mapper.enhance_query(query)
        
        # Ensure enhanced_query is a string
        if isinstance(enhanced_query, list):
            enhanced_query = ' '.join(str(item) for item in enhanced_query) if enhanced_query else query
        elif not isinstance(enhanced_query, str):
            enhanced_query = str(enhanced_query) if enhanced_query else query
        
        # Fall back to original query if enhanced query is empty
        if not enhanced_query.strip():
            enhanced_query = query
        
        # Step 2: Search database with enhanced query (with chunk caching)
        try:
            # Generate embedding for the enhanced query
            enhanced_embedding = None
            if self.enable_cache and self.chunk_cache:
                try:
                    enhanced_embedding = np.array(self.embedding_function.embed_query(enhanced_query))
                    
                    # Try to get cached chunks
                    cached_chunks = self.chunk_cache.get(enhanced_embedding, k)
                    if cached_chunks is not None:
                        print(f"✅ Chunk cache hit")
                        results = cached_chunks
                    else:
                        # Perform similarity search
                        results = self.db.similarity_search(enhanced_query, k=k)
                        # Cache the chunks
                        self.chunk_cache.set(enhanced_embedding, k, results)
                except Exception as e:
                    print(f"Warning: Chunk cache error: {e}")
                    results = self.db.similarity_search(enhanced_query, k=k)
            else:
                results = self.db.similarity_search(enhanced_query, k=k)
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
        
        # Step 3: Collect related images and prepare context
        related_images = []
        context_parts = []
        
        for doc in results:
            context_parts.append(doc.page_content)
            images = self._collect_images_from_result(doc)
            related_images.extend(images)
        
        # Remove duplicate images based on filename
        seen_filenames = set()
        unique_images = []
        for img in related_images:
            if img['filename'] not in seen_filenames:
                unique_images.append(img)
                seen_filenames.add(img['filename'])
        
        # Step 4: Generate game piece context
        game_piece_context = ""
        if matched_pieces:
            game_piece_context = self.game_piece_mapper.get_context_for_pieces(matched_pieces)

        # Step 5: Generate AI response
        context_text = "\n\n---\n\n".join(context_parts)
        
        try:
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(
                context=context_text,
                game_piece_context=game_piece_context,
                question=query
            )
            
            model = Ollama(model="gpt-oss:20b")
            response_text = model.invoke(prompt)
            
        except Exception as e:
            # Fallback response if AI generation fails
            response_text = f"Based on the technical documentation, here's what I found:\n\n{context_text[:1000]}..."
            if game_piece_context:
                response_text += f"\n\nGame Piece Information:\n{game_piece_context}"

        response = {
            "response": response_text,
            "matched_pieces": matched_pieces,
            "enhanced_query": enhanced_query,
            "original_query": query,
            "context_sources": len(results),
            "related_images": unique_images,
            "game_piece_context": game_piece_context
        }
        
        # Cache the response
        if self.enable_cache and self.query_cache:
            try:
                query_embedding = np.array(self.embedding_function.embed_query(query))
                self.query_cache.set(query, response, k, query_embedding)
                print("✅ Response cached")
            except Exception as e:
                print(f"Warning: Could not cache response: {e}")
        
        return response

    def _collect_images_from_result(self, doc) -> List[Dict[str, Any]]:
        """Collect image information from a document result"""
        images_info = []
        
        try:
            # Check if this is an image context document (preferred)
            if doc.metadata.get('type') == 'image_context':
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
            elif doc.metadata.get('type') == 'image_text':
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
            elif doc.metadata.get('type') == 'text_with_images':
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

    def search_by_season(self, season: str, query: str = "", k: int = 5) -> Dict[str, Any]:
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
    def prepare_query_metadata(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Prepare metadata for a query (images, matched pieces, etc.) without generating the response.
        This is used for streaming to send metadata first.
        """
        # Step 1: Enhance query with game piece context
        matched_pieces, enhanced_query = self.game_piece_mapper.enhance_query(query)
        
        # Ensure enhanced_query is a string
        if isinstance(enhanced_query, list):
            enhanced_query = ' '.join(str(item) for item in enhanced_query) if enhanced_query else query
        elif not isinstance(enhanced_query, str):
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
                        results = self.db.similarity_search(enhanced_query, k=k)
                        self.chunk_cache.set(enhanced_embedding, k, results)
                except Exception as e:
                    results = self.db.similarity_search(enhanced_query, k=k)
            else:
                results = self.db.similarity_search(enhanced_query, k=k)
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
                "context_parts": []
            }
        
        # Step 3: Collect related images and prepare context
        related_images = []
        context_parts = []
        
        for doc in results:
            context_parts.append(doc.page_content)
            images = self._collect_images_from_result(doc)
            related_images.extend(images)
        
        # Remove duplicate images
        seen_filenames = set()
        unique_images = []
        for img in related_images:
            if img['filename'] not in seen_filenames:
                unique_images.append(img)
                seen_filenames.add(img['filename'])
        
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
                'context_summary': img.get('context_summary', img.get('ocr_text', '')[:200] + ('...' if len(img.get('ocr_text', '')) > 200 else ''))
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
            "context_parts": context_parts
        }
    
    def stream_query_response(self, query: str, metadata: Dict[str, Any]):
        """
        Stream the LLM response for a query.
        Yields chunks of text as they're generated.
        """
        context_text = "\n\n---\n\n".join(metadata['context_parts'])
        game_piece_context = metadata.get('game_piece_context', '')
        
        try:
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(
                context=context_text,
                game_piece_context=game_piece_context,
                question=query
            )
            
            # Use Ollama with streaming
            model = Ollama(model="gpt-oss:20b")
            
            # Stream the response
            for chunk in model.stream(prompt):
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
