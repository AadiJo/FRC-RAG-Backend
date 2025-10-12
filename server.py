"""
Main FRC RAG Server Application
Production-ready server with rate limiting, tunneling, and monitoring
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import TooManyRequests

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.server.config import get_config
from src.server.ollama_proxy import OllamaProxy
from src.server.tunnel import TunnelManager
from src.core.query_processor import QueryProcessor

Config = get_config()

# Configure logging
os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize components
ollama_proxy = OllamaProxy()
tunnel_manager = TunnelManager()
query_processor = None

def init_query_processor():
    """Initialize the query processor"""
    global query_processor
    try:
        query_processor = QueryProcessor(Config.CHROMA_PATH, Config.IMAGES_PATH)
        logger.info("Query processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing query processor: {e}")
        return False

def require_api_key():
    """Check API key if required"""
    if Config.API_KEY_REQUIRED:
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in Config.VALID_API_KEYS:
            return jsonify({"error": "Valid API key required"}), 401
    return None

@app.errorhandler(TooManyRequests)
def handle_rate_limit(e):
    """Handle rate limit exceptions"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "timestamp": datetime.now().isoformat()
    }), 429

@app.before_request
def log_request():
    """Log incoming requests"""
    if Config.LOG_LEVEL == 'DEBUG':
        logger.debug(f"{request.method} {request.path} from {request.remote_addr}")

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    ollama_healthy = ollama_proxy.check_health()
    stats = ollama_proxy.get_stats()
    tunnel_status = tunnel_manager.get_status()
    
    # Get cache stats if available
    cache_stats = None
    if query_processor:
        try:
            cache_stats = query_processor.get_cache_stats()
        except Exception as e:
            logger.warning(f"Could not get cache stats: {e}")
    
    status = "healthy" if ollama_healthy and query_processor else "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ollama": ollama_healthy,
            "query_processor": query_processor is not None,
            "chroma_db": os.path.exists(Config.CHROMA_PATH),
            "images": os.path.exists(Config.IMAGES_PATH)
        },
        "stats": stats,
        "cache": cache_stats,
        "tunnel": tunnel_status,
        "config": {
            "environment": Config.ENVIRONMENT,
            "rate_limit": f"{Config.RATE_LIMIT_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW} minute(s)",
            "api_key_required": Config.API_KEY_REQUIRED,
            "tunnel_configured": bool(Config.TUNNEL_SERVICE)
        }
    })

@app.route('/api/query', methods=['POST'])
def api_query():
    """Rate-limited query endpoint"""
    try:
        # Check API key if required
        auth_error = require_api_key()
        if auth_error:
            return auth_error
        
        client_id = ollama_proxy.get_client_id(request)
        
        # Check rate limit
        if not ollama_proxy.rate_limiter.is_allowed(client_id):
            client_stats = ollama_proxy.rate_limiter.get_client_stats(client_id)
            return jsonify({
                "error": "Rate limit exceeded",
                "rate_limit": client_stats,
                "timestamp": datetime.now().isoformat()
            }), 429
        
        # Process request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        k = data.get('k', 5)
        result = query_processor.process_query(query_text, k)
        
        # Log the query (without sensitive data)
        logger.info(f"Query processed for client {client_id}: {len(query_text)} chars")
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Process images for web display
        web_images = []
        for img in result.get("related_images", []):
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
        
        client_stats = ollama_proxy.rate_limiter.get_client_stats(client_id)
        
        return jsonify({
            "success": True,
            "query": result["original_query"],
            "enhanced_query": result["enhanced_query"],
            "response": result["response"],
            "images": web_images,
            "results_count": result["context_sources"],
            "images_count": len(web_images),
            "matched_pieces": result["matched_pieces"],
            "game_piece_context": result["game_piece_context"],
            "rate_limit": client_stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        logger.error(f"Error in api_query: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/query/stream', methods=['POST'])
def api_query_stream():
    """Streaming query endpoint for real-time response generation"""
    try:
        # Check API key if required
        auth_error = require_api_key()
        if auth_error:
            return auth_error
        
        client_id = ollama_proxy.get_client_id(request)
        
        # Check rate limit
        if not ollama_proxy.rate_limiter.is_allowed(client_id):
            client_stats = ollama_proxy.rate_limiter.get_client_stats(client_id)
            return jsonify({
                "error": "Rate limit exceeded",
                "rate_limit": client_stats,
                "timestamp": datetime.now().isoformat()
            }), 429
        
        # Process request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        k = data.get('k', 5)
        
        def generate():
            """Generator function for streaming responses"""
            try:
                # First, send metadata (images, matched pieces, etc.)
                metadata = query_processor.prepare_query_metadata(query_text, k)
                
                if "error" in metadata:
                    yield f"data: {json.dumps({'error': metadata['error']})}\n\n"
                    return
                
                # Send metadata event
                yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
                
                # Stream the response
                for chunk in query_processor.stream_query_response(query_text, metadata):
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                
                # Send completion event
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream generation: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        logger.error(f"Error in api_query_stream: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/ollama/<path:path>', methods=['GET', 'POST'])
def proxy_ollama(path):
    """Proxy endpoint for direct Ollama access with rate limiting"""
    try:
        # Check API key if required
        auth_error = require_api_key()
        if auth_error:
            return auth_error
        
        client_id = ollama_proxy.get_client_id(request)
        
        json_data = request.get_json() if request.method == 'POST' else None
        result = ollama_proxy.proxy_request(client_id, f"api/{path}", request.method, json_data)
        
        if "error" in result:
            return jsonify(result), 500 if result.get("status") != "service_unavailable" else 503
        
        return jsonify(result)
        
    except TooManyRequests as e:
        return jsonify({
            "error": "Rate limit exceeded",
            "message": str(e.description),
            "timestamp": datetime.now().isoformat()
        }), 429
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        logger.error(f"Error in ollama proxy: {e}", exc_info=True)
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

@app.route('/api/stats')
def api_stats():
    """Get comprehensive server statistics"""
    return jsonify({
        **ollama_proxy.get_stats(),
        "config": Config.to_dict(),
        "tunnel": tunnel_manager.get_status(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/tunnel', methods=['POST'])
def api_tunnel():
    """Start/stop tunnel service"""
    try:
        # Check API key if required
        auth_error = require_api_key()
        if auth_error:
            return auth_error
        
        data = request.get_json()
        action = data.get('action', 'status')
        
        if action == 'start':
            url = tunnel_manager.start_tunnel()
            return jsonify({
                "action": "start",
                "url": url,
                "status": tunnel_manager.get_status()
            })
        elif action == 'stop':
            tunnel_manager.stop_tunnel()
            return jsonify({
                "action": "stop",
                "status": tunnel_manager.get_status()
            })
        else:
            return jsonify({
                "action": "status",
                "status": tunnel_manager.get_status()
            })
            
    except Exception as e:
        logger.error(f"Error in tunnel management: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    """Serve images from the data/images directory"""
    return send_from_directory(Config.IMAGES_PATH, filepath)

@app.route('/api/suggestions', methods=['POST'])
def api_suggestions():
    """API endpoint for getting game piece suggestions"""
    try:
        client_id = ollama_proxy.get_client_id(request)
        
        # Light rate limiting for suggestions
        if not ollama_proxy.rate_limiter.is_allowed(client_id):
            return jsonify({"suggestions": []}), 429
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        data = request.get_json()
        partial_query = data.get('query', '').strip()
        
        if not partial_query:
            return jsonify({"suggestions": []})
        
        suggestions = query_processor.get_game_piece_suggestions(partial_query)
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        logger.error(f"Error in suggestions: {e}", exc_info=True)
        return jsonify({"suggestions": []}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for receiving user feedback on responses"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('query', '').strip()
        response_text = data.get('response', '').strip()
        feedback_type = data.get('feedback_type', '').strip()  # 'good', 'bad', 'redo'
        timestamp = datetime.now().isoformat()
        
        if not query or not feedback_type:
            return jsonify({"error": "Query and feedback_type are required"}), 400
        
        if feedback_type not in ['good', 'bad', 'redo']:
            return jsonify({"error": "feedback_type must be 'good', 'bad', or 'redo'"}), 400
        
        client_id = ollama_proxy.get_client_id(request)
        
        # Log the feedback
        feedback_log = {
            "timestamp": timestamp,
            "client_id": client_id,
            "query": query,
            "response_preview": response_text[:200] + ('...' if len(response_text) > 200 else ''),
            "feedback_type": feedback_type,
            "ip_address": request.remote_addr,
            "user_agent": request.headers.get('User-Agent', '')
        }
        
        logger.info(f"FEEDBACK: {feedback_type.upper()} - Client: {client_id} - Query: {query[:100]}")
        logger.info(f"FEEDBACK_DETAIL: {json.dumps(feedback_log)}")
        
        return jsonify({
            "success": True,
            "message": "Feedback received successfully",
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({"error": "Failed to process feedback"}), 500

@app.route('/api/seasons')
def api_seasons():
    """API endpoint for getting available seasons"""
    try:
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        seasons = query_processor.get_available_seasons()
        return jsonify({
            "seasons": seasons,
            "count": len(seasons)
        })
    except Exception as e:
        logger.error(f"Error getting seasons: {e}", exc_info=True)
        return jsonify({"error": "Failed to get seasons"}), 500

@app.route('/api/cache/stats')
def api_cache_stats():
    """API endpoint for getting cache statistics"""
    try:
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        cache_stats = query_processor.get_cache_stats()
        return jsonify(cache_stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        return jsonify({"error": "Failed to get cache stats"}), 500

@app.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """API endpoint for clearing the cache (admin only)"""
    try:
        # Check API key if required
        key_error = require_api_key()
        if key_error:
            return key_error
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        query_processor.clear_cache()
        logger.info("Cache cleared via API")
        
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({"error": "Failed to clear cache"}), 500

@app.route('/api/cache/reset-stats', methods=['POST'])
def api_cache_reset_stats():
    """API endpoint for resetting cache statistics"""
    try:
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        query_processor.reset_cache_stats()
        logger.info("Cache stats reset via API")
        
        return jsonify({
            "success": True,
            "message": "Cache statistics reset successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error resetting cache stats: {e}", exc_info=True)
        return jsonify({"error": "Failed to reset cache stats"}), 500

def cleanup_on_exit():
    """Cleanup function called on exit"""
    logger.info("Shutting down server...")
    tunnel_manager.stop_tunnel()
    ollama_proxy.cleanup()

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting FRC RAG Server")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Host: {Config.HOST}:{Config.PORT}")
    logger.info(f"Ollama: {Config.get_ollama_url()}")
    logger.info(f"Rate limiting: {Config.RATE_LIMIT_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW} minute(s)")
    logger.info("="*60)
    
    # Initialize query processor
    if init_query_processor():
        logger.info("✓ Query processor initialized")
    else:
        logger.error("✗ Query processor initialization failed")
    
    # Check Ollama health
    if ollama_proxy.check_health():
        logger.info("✓ Ollama service is healthy")
    else:
        logger.warning("✗ Ollama service is not available")
    
    # Start tunnel if configured
    if Config.TUNNEL_SERVICE:
        logger.info(f"Starting {Config.TUNNEL_SERVICE} tunnel...")
        tunnel_url = tunnel_manager.start_tunnel()
        if tunnel_url:
            logger.info(f"✓ Tunnel URL: {tunnel_url}")
        else:
            logger.warning("✗ Failed to start tunnel")
    
    try:
        app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        cleanup_on_exit()