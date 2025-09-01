import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from typing import List, Dict, Any
from enhanced_query_processor import EnhancedQueryProcessor

app = Flask(__name__)

# Configuration
CHROMA_PATH = "db"  # Updated to match create_database.py
IMAGES_PATH = "data/images"

# Initialize enhanced query processor
query_processor = None

def init_query_processor():
    """Initialize the enhanced query processor"""
    global query_processor
    try:
        query_processor = EnhancedQueryProcessor(CHROMA_PATH, IMAGES_PATH)
        return True
    except Exception as e:
        print(f"Error initializing query processor: {e}")
        return False

def query_database(query_text: str, k: int = 5) -> Dict[str, Any]:
    """Query the database using enhanced query processor"""
    if not query_processor:
        return {"error": "Query processor not initialized. Please run create_database.py first."}
    
    # Use enhanced query processor
    result = query_processor.process_query(query_text, k)
    
    if "error" in result:
        return {"error": result["error"]}
    
    # Process images for web display
    web_images = []
    for img in result.get("related_images", []):
        # Check if image file exists
        img_exists = os.path.exists(img['file_path'])
        # Create web-accessible path
        web_path = img['file_path'].replace(IMAGES_PATH + '/', '')
        
        web_images.append({
            'filename': img['filename'],
            'file_path': img['file_path'],
            'web_path': web_path,
            'page': img.get('page'),
            'exists': img_exists,
            'ocr_text': img.get('ocr_text', '')
        })
    
    return {
        "success": True,
        "query": result["original_query"],
        "enhanced_query": result["enhanced_query"],
        "response": result["response"],
        "images": web_images,
        "results_count": result["context_sources"],
        "images_count": len(web_images),
        "matched_pieces": result["matched_pieces"],
        "game_piece_context": result["game_piece_context"]
    }

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries"""
    data = request.get_json()
    query_text = data.get('query', '').strip()
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    k = data.get('k', 5)
    result = query_database(query_text, k)
    
    return jsonify(result)

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    """Serve images from the data/images directory"""
    return send_from_directory(IMAGES_PATH, filepath)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "query_processor_initialized": query_processor is not None,
        "chroma_path_exists": os.path.exists(CHROMA_PATH),
        "images_path_exists": os.path.exists(IMAGES_PATH)
    })

@app.route('/api/suggestions', methods=['POST'])
def api_suggestions():
    """API endpoint for getting game piece suggestions"""
    if not query_processor:
        return jsonify({"error": "Query processor not initialized"}), 500
    
    data = request.get_json()
    partial_query = data.get('query', '').strip()
    
    if not partial_query:
        return jsonify({"suggestions": []})
    
    suggestions = query_processor.get_game_piece_suggestions(partial_query)
    return jsonify({"suggestions": suggestions})

@app.route('/api/seasons')
def api_seasons():
    """API endpoint for getting available seasons"""
    if not query_processor:
        return jsonify({"error": "Query processor not initialized"}), 500
    
    seasons = query_processor.get_available_seasons()
    return jsonify({"seasons": seasons})

if __name__ == '__main__':
    print("Initializing FRC RAG Chat Interface with Enhanced Game Piece Mapping...")
    
    # Initialize query processor
    if init_query_processor():
        print("Enhanced query processor initialized successfully")
    else:
        print("Query processor initialization failed")
    
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
