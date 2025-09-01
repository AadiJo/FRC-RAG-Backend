"""
Enhanced Query Processor - Integrates game piece mapping with RAG system
This module processes user queries and modifies them with game piece context
"""

import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from game_piece_mapper import GamePieceMapper

class EnhancedQueryProcessor:
    def __init__(self, chroma_path: str = "db", images_path: str = "data/images"):
        self.chroma_path = chroma_path
        self.images_path = images_path
        self.game_piece_mapper = GamePieceMapper()
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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

Answer:
"""

    def _init_database(self):
        """Initialize the ChromaDB connection"""
        if not os.path.exists(self.chroma_path):
            print(f"Database not found at {self.chroma_path}")
            return False
        
        try:
            self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
            return True
        except Exception as e:
            print(f"Error initializing database: {e}")
            return False

    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query with game piece enhancement
        Returns a comprehensive response with context and metadata
        """
        if not self.db:
            return {
                "error": "Database not initialized",
                "response": "Sorry, the database is not available.",
                "matched_pieces": [],
                "enhanced_query": query
            }

        # Step 1: Game piece mapping
        enhanced_query, matched_pieces = self.game_piece_mapper.enhance_query(query)
        game_piece_context = self.game_piece_mapper.get_context_for_pieces(matched_pieces)
        
        # Step 2: Search the database with enhanced query
        try:
            results = self.db.similarity_search_with_relevance_scores(enhanced_query, k=k)
            
            if not results or results[0][1] < 0.1:
                # Try original query if enhanced query doesn't work well
                results = self.db.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            return {
                "error": f"Database search error: {e}",
                "response": "Sorry, there was an error searching the database.",
                "matched_pieces": matched_pieces,
                "enhanced_query": enhanced_query
            }

        if not results or results[0][1] < 0.1:
            return {
                "error": "No relevant results found",
                "response": f"I couldn't find relevant information for '{query}'. Try rephrasing your question or being more specific.",
                "matched_pieces": matched_pieces,
                "enhanced_query": enhanced_query
            }

        # Step 3: Process results and collect context
        context_parts = []
        related_images = []
        
        for i, (doc, score) in enumerate(results, 1):
            # Add document content to context
            context_parts.append(f"[Source {i} - Page {doc.metadata.get('page', 'N/A')} - Score: {score:.3f}]:\n{doc.page_content}")
            
            # Collect image information
            images_info = self._collect_images_from_result(doc)
            related_images.extend(images_info)

        # Step 4: Generate AI response
        context_text = "\n\n---\n\n".join(context_parts)
        
        try:
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(
                context=context_text,
                game_piece_context=game_piece_context,
                question=query
            )
            
            model = Ollama(model="mistral")
            response_text = model.invoke(prompt)
            
        except Exception as e:
            # Fallback response if AI generation fails
            response_text = f"Based on the technical documentation, here's what I found:\n\n{context_text[:1000]}..."
            if game_piece_context:
                response_text += f"\n\nGame Piece Information:\n{game_piece_context}"

        return {
            "response": response_text,
            "matched_pieces": matched_pieces,
            "enhanced_query": enhanced_query,
            "original_query": query,
            "context_sources": len(results),
            "related_images": related_images,
            "game_piece_context": game_piece_context
        }

    def _collect_images_from_result(self, doc) -> List[Dict[str, Any]]:
        """Collect image information from a document result"""
        images_info = []
        
        # Check if this is an image document
        if doc.metadata.get('type') == 'image_text':
            image_file = doc.metadata.get('image_file')
            image_path = doc.metadata.get('image_path')
            if image_file and image_path:
                images_info.append({
                    'filename': image_file,
                    'file_path': image_path,
                    'page': doc.metadata.get('page'),
                    'ocr_text': doc.page_content.replace('Image content: ', '').split('\n\nContext:')[0]
                })
        
        # Check for text documents with associated images
        elif doc.metadata.get('type') == 'text_with_images':
            import json
            image_filenames_str = doc.metadata.get('image_filenames', '[]')
            try:
                image_filenames = json.loads(image_filenames_str)
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
                        'ocr_text': ''
                    })
            except json.JSONDecodeError:
                pass
        
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
                    "game": piece_data["game"]
                })
            
            # Check generic names
            for generic_name in piece_data["generic_names"]:
                if partial_lower in generic_name.lower():
                    suggestions.append({
                        "type": "generic_name", 
                        "text": generic_name,
                        "maps_to": piece_data["official_name"],
                        "season": piece_data["season"]
                    })
            
            # Check synonyms
            for synonym in piece_data["synonyms"]:
                if partial_lower in synonym.lower():
                    suggestions.append({
                        "type": "synonym",
                        "text": synonym,
                        "maps_to": piece_data["official_name"],
                        "season": piece_data["season"]
                    })
        
        return suggestions[:10]  # Limit to 10 suggestions

    def search_by_season(self, season: str, query: str = "", k: int = 5) -> Dict[str, Any]:
        """
        Search for information specific to a season
        """
        # Get game pieces for the season
        season_pieces = self.game_piece_mapper.get_pieces_by_season(season)
        
        # Season-specific context
        season_context = f"season {season} "
        for piece_id in season_pieces:
            piece_data = self.game_piece_mapper.get_piece_info(piece_id)
            season_context += f"{piece_data.get('official_name', '')} "
        
        enhanced_query = f"{query} {season_context}".strip()
        
        return self.process_query(enhanced_query, k)

    def get_available_seasons(self) -> List[str]:
        """Get list of all available game seasons"""
        return self.game_piece_mapper.get_all_seasons()

# Example usage
if __name__ == "__main__":
    processor = EnhancedQueryProcessor()
    
    test_queries = [
        "How do I pick up a ball?",
        "What's the best intake for cubes?", 
        "Show me speaker scoring mechanisms",
        "How to handle rings in 2024?",
        "What are cone scoring strategies?"
    ]
    
    print("Testing Enhanced Query Processor")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nProcessing: {query}")
        result = processor.process_query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Enhanced query: {result['enhanced_query']}")
            print(f"Matched pieces: {result['matched_pieces']}")
            print(f"Response length: {len(result['response'])} characters")
            if result['related_images']:
                print(f"Related images: {len(result['related_images'])}")
        print("-" * 30)
