import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = Flask(__name__)

# Configuration
CHROMA_PATH = "db"  # Updated to match create_database.py
IMAGES_PATH = "data/images"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

If relevant images are mentioned in the context, include references to them in your answer.
Based on the context and your knowledge, provide a detailed and accurate response, and draw conclusions if applicable.
If the context does not provide enough information for the full answer, connect what is provided with your own knowledge to give a comprehensive response.
"""

# Initialize embedding function and database
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = None

def init_database():
    """Initialize the database connection"""
    global db
    if not os.path.exists(CHROMA_PATH):
        print(f"Database not found at {CHROMA_PATH}. Please run create_database.py first.")
        return False
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return True

def collect_images_from_result(doc) -> List[Dict[str, Any]]:
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
    
    # Check if this is an image metadata document
    elif doc.metadata.get('type') == 'image_info':
        image_file = doc.metadata.get('image_file')
        image_path = doc.metadata.get('image_path')
        if image_file and image_path:
            images_info.append({
                'filename': image_file,
                'file_path': image_path,
                'page': doc.metadata.get('page'),
                'ocr_text': ''
            })
    
    # Check if this document has associated images
    elif doc.metadata.get('type') == 'text_with_images':
        image_filenames_str = doc.metadata.get('image_filenames', '[]')
        try:
            image_filenames = json.loads(image_filenames_str)
            page_num = doc.metadata.get('page', 1)
            
            # Determine PDF name from source path
            source_path = doc.metadata.get('source', '')
            pdf_name = os.path.splitext(os.path.basename(source_path))[0] if source_path else ''
            
            for filename in image_filenames:
                # Construct path using PDF subfolder
                if pdf_name:
                    pdf_subfolder_path = os.path.join(IMAGES_PATH, pdf_name, filename)
                else:
                    pdf_subfolder_path = None
                
                legacy_path = os.path.join(IMAGES_PATH, filename)
                
                # Check both new and old paths for compatibility
                if pdf_subfolder_path and os.path.exists(pdf_subfolder_path):
                    file_path = pdf_subfolder_path
                elif os.path.exists(legacy_path):
                    file_path = legacy_path
                else:
                    file_path = pdf_subfolder_path if pdf_subfolder_path else legacy_path
                
                images_info.append({
                    'filename': filename,
                    'file_path': file_path,
                    'page': page_num,
                    'ocr_text': ''
                })
        except:
            pass  # If JSON parsing fails, continue without images
    
    return images_info

def query_database(query_text: str, k: int = 5) -> Dict[str, Any]:
    """Query the database and return results with images"""
    if not db:
        return {"error": "Database not initialized. Please run create_database.py first."}
    
    # Search database
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    
    if len(results) == 0 or results[0][1] < 0.1:
        return {"error": f"No relevant results found for: '{query_text}'"}
    
    # Process results and collect related images
    context_parts = []
    related_images = []
    
    for i, (doc, score) in enumerate(results, 1):
        # Add to context
        context_parts.append(f"[Result {i} from page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content}")
        
        # Collect images
        images_info = collect_images_from_result(doc)
        if images_info:
            related_images.extend(images_info)
    
    # Generate response using Ollama
    context_text = "\n\n---\n\n".join(context_parts)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    try:
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
    except Exception as e:
        print(f"Error generating AI response: {e}")
        response_text = "Error generating AI response. Showing direct context instead."
    
    # Collect unique images
    unique_images = {}
    for img in related_images:
        if img['filename'] not in unique_images:
            # Check if image file exists
            img['exists'] = os.path.exists(img['file_path'])
            # Create web-accessible path
            img['web_path'] = img['file_path'].replace(IMAGES_PATH + '/', '')
            unique_images[img['filename']] = img
    
    return {
        "success": True,
        "query": query_text,
        "response": response_text,
        "images": list(unique_images.values()),
        "results_count": len(results),
        "images_count": len(unique_images)
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
        "database_initialized": db is not None,
        "chroma_path_exists": os.path.exists(CHROMA_PATH),
        "images_path_exists": os.path.exists(IMAGES_PATH)
    })

if __name__ == '__main__':
    print("Initializing FRC RAG Chat Interface...")
    
    # Initialize database
    if init_database():
        print("Database initialized successfully")
    else:
        print("Database initialization failed")
    
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
