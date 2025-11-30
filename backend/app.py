"""
Main FRC RAG Server Application
Production-ready server with rate limiting, tunneling, and monitoring
"""

import os
# Set environment variables to suppress TensorFlow and ChromaDB noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import TooManyRequests

from src.server.config import get_config
from src.server.ollama_proxy import OllamaProxy
from src.server.tunnel import TunnelManager
from src.core.query_processor import QueryProcessor
from src.utils.feedback_manager import FeedbackManager

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

# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize components
ollama_proxy = OllamaProxy()
tunnel_manager = TunnelManager()
query_processor = None
feedback_manager = FeedbackManager(os.path.join(os.path.dirname(__file__), 'data'))

def init_query_processor():
    """Initialize the query processor"""
    global query_processor
    try:
        query_processor = QueryProcessor(Config.CHROMA_PATH, Config.IMAGES_PATH)
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

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    ollama_healthy = ollama_proxy.check_health() if Config.MODEL_PROVIDER == 'local' else True
    chutes_healthy = False
    if Config.MODEL_PROVIDER == 'chute' and query_processor and hasattr(query_processor, 'chutes_client') and query_processor.chutes_client:
        chutes_healthy = query_processor.chutes_client.check_health()
    
    stats = ollama_proxy.get_stats()
    tunnel_status = tunnel_manager.get_status()
    
    # Get cache stats if available
    cache_stats = None
    if query_processor:
        try:
            cache_stats = query_processor.get_cache_stats()
        except Exception as e:
            logger.warning(f"Could not get cache stats: {e}")
    
    # Determine overall status based on provider
    if Config.MODEL_PROVIDER == 'chute':
        model_healthy = chutes_healthy
    else:
        model_healthy = ollama_healthy
        
    status = "healthy" if model_healthy and query_processor else "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ollama": ollama_healthy if Config.MODEL_PROVIDER == 'local' else None,
            "chutes_ai": chutes_healthy if Config.MODEL_PROVIDER == 'chute' else None,
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
        conversation_history = data.get('conversation_history', [])
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        k = data.get('k', 10)
        enable_filtering = data.get('enable_filtering', False)  # Only enable if requested
        result = query_processor.process_query(query_text, k, enable_filtering=enable_filtering, 
                                               conversation_history=conversation_history)
        
        # Log the query (without sensitive data)
        logger.info(f"Query processed for client {client_id}: {len(query_text)} chars")
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Process images for web display
        web_images = []
        for img in result.get("related_images", []):
            img_exists = os.path.exists(img['file_path'])
            try:
                # Use relpath for safer path calculation
                web_path = os.path.relpath(img['file_path'], Config.IMAGES_PATH)
                # Ensure forward slashes for URLs
                web_path = web_path.replace(os.sep, '/')
            except ValueError:
                # Fallback if paths are on different drives or incompatible
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
        conversation_history = data.get('conversation_history', [])
        show_reasoning = data.get('show_reasoning', None)
        custom_api_key = data.get('custom_api_key', None)
        custom_model = data.get('custom_model', None)
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        if not query_processor:
            return jsonify({"error": "Query processor not initialized"}), 500
        
        k = data.get('k', 10)
        
        def generate():
            """Generator function for streaming responses"""
            try:
                # First, send metadata (images, matched pieces, etc.)
                enable_filtering = data.get('enable_filtering', False)  # Only enable if requested
                metadata = query_processor.prepare_query_metadata(query_text, k, enable_filtering=enable_filtering,
                                                                  conversation_history=conversation_history)
                
                if "error" in metadata:
                    yield f"data: {json.dumps({'error': metadata['error']})}\n\n"
                    return
                
                # Send metadata event
                yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
                
                # Stream the response
                for chunk in query_processor.stream_query_response(
                    query_text, 
                    metadata, 
                    show_reasoning=show_reasoning,
                    custom_api_key=custom_api_key,
                    custom_model=custom_model
                ):
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
            "response_full": response_text,
            "response_preview": response_text[:200] + ('...' if len(response_text) > 200 else ''),
            "feedback_type": feedback_type,
            "ip_address": request.remote_addr,
            "user_agent": request.headers.get('User-Agent', '')
        }
        
        # Save to file
        feedback_manager.save_feedback(feedback_log)
        
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

@app.route('/api/chutes/validate', methods=['POST'])
def api_validate_chutes_key():
    """API endpoint for validating a Chutes API key"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        api_key = data.get('api_key', '').strip()
        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        
        # Test the API key by making a simple request to a free model
        import requests as http_requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use a free model for testing
        test_data = {
            "model": "unsloth/gemma-3-4b-it",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": False
        }
        
        response = http_requests.post(
            "https://llm.chutes.ai/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=15,
            verify=False
        )
        
        if response.status_code == 200:
            return jsonify({
                "valid": True,
                "message": "API key is valid"
            })
        elif response.status_code == 401:
            return jsonify({
                "valid": False,
                "message": "Invalid API key"
            })
        else:
            return jsonify({
                "valid": False,
                "message": f"API validation failed: {response.status_code}"
            })
            
    except http_requests.exceptions.Timeout:
        return jsonify({
            "valid": False,
            "message": "Request timed out - try again"
        })
    except Exception as e:
        logger.error(f"Error validating Chutes API key: {e}", exc_info=True)
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500

@app.route('/api/chutes/models')
def api_get_chutes_models():
    """API endpoint for getting available Chutes models"""
    try:
        # List of popular LLM models available on Chutes (verified model IDs)
        # These are the text generation models (LLM type) that work with chat completions
        models = [
            # Free models
            {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "free": True},
            {"id": "unsloth/gemma-3-4b-it", "name": "Gemma 3 4B", "free": True},
            {"id": "zai-org/GLM-4.5-Air", "name": "GLM 4.5 Air", "free": True},
            {"id": "meituan-longcat/LongCat-Flash-Chat-FP8", "name": "LongCat Flash Chat", "free": True},
            {"id": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", "name": "Tongyi DeepResearch 30B", "free": True},
            # Paid models
            {"id": "deepseek-ai/DeepSeek-V3.2-Exp", "name": "DeepSeek V3.2 Exp", "free": False},
            {"id": "deepseek-ai/DeepSeek-R1-0528", "name": "DeepSeek R1 0528", "free": False},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "free": False},
            {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek V3", "free": False},
            {"id": "deepseek-ai/DeepSeek-V3.1", "name": "DeepSeek V3.1", "free": False},
            {"id": "deepseek-ai/DeepSeek-V3.1-Terminus", "name": "DeepSeek V3.1 Terminus", "free": False},
            {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B Instruct", "free": False},
            {"id": "Qwen/Qwen3-32B", "name": "Qwen3 32B", "free": False},
            {"id": "Qwen/Qwen3-14B", "name": "Qwen3 14B", "free": False},
            {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "name": "Qwen3 235B Instruct", "free": False},
            {"id": "Qwen/Qwen3-Next-80B-A3B-Instruct", "name": "Qwen3 Next 80B Instruct", "free": False},
            {"id": "tngtech/DeepSeek-TNG-R1T2-Chimera", "name": "DeepSeek TNG R1T2 Chimera", "free": False},
            {"id": "zai-org/GLM-4.6", "name": "GLM 4.6", "free": False},
            {"id": "zai-org/GLM-4.5", "name": "GLM 4.5", "free": False},
            {"id": "chutesai/Mistral-Small-3.1-24B-Instruct-2503", "name": "Mistral Small 3.1 24B", "free": False},
            {"id": "chutesai/Mistral-Small-3.2-24B-Instruct-2506", "name": "Mistral Small 3.2 24B", "free": False},
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "free": False},
            {"id": "moonshotai/Kimi-K2-Instruct-0905", "name": "Kimi K2 Instruct", "free": False},
            {"id": "NousResearch/Hermes-4-405B-FP8", "name": "Hermes 4 405B", "free": False},
            {"id": "NousResearch/DeepHermes-3-Mistral-24B-Preview", "name": "DeepHermes 3 Mistral 24B", "free": False},
            {"id": "unsloth/gemma-3-12b-it", "name": "Gemma 3 12B", "free": False},
            {"id": "unsloth/gemma-3-27b-it", "name": "Gemma 3 27B", "free": False},
        ]
        
        return jsonify({
            "models": models,
            "default_model": "openai/gpt-oss-20b"
        })
        
    except Exception as e:
        logger.error(f"Error getting Chutes models: {e}", exc_info=True)
        return jsonify({"error": "Failed to get models"}), 500

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
    if Config.MODEL_PROVIDER == 'local':
        logger.info(f"Ollama: {Config.get_ollama_url()}")
    logger.info("="*60)
    
    # Initialize query processor
    if init_query_processor():
        logger.info("✓ Query processor initialized")
    else:
        logger.error("✗ Query processor initialization failed")
    
    # Check Ollama health only if local
    if Config.MODEL_PROVIDER == 'local':
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