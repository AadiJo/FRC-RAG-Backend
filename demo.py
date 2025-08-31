#!/usr/bin/env python3
"""
FRC RAG Chat Interface Demo
Demonstrates the capabilities of the web interface with example queries
"""

import os
import time
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"
DEMO_QUERIES = [
    {
        "query": "tube shaped object",
        "description": "Search for cylindrical components and tubes"
    },
    {
        "query": "coral picker mechanism", 
        "description": "Find coral manipulation mechanisms"
    },
    {
        "query": "CAD design",
        "description": "Locate CAD drawings and technical designs"
    },
    {
        "query": "gripper mechanism",
        "description": "Search for gripping and manipulation systems"
    },
    {
        "query": "elevator design",
        "description": "Find lifting and elevator mechanisms"
    }
]

def check_server():
    """Check if the web server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server is running")
            print(f"   Database initialized: {health_data.get('database_initialized', False)}")
            print(f"   ChromaDB exists: {health_data.get('chroma_path_exists', False)}")
            print(f"   Images path exists: {health_data.get('images_path_exists', False)}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the server is running with: python app.py")
        return False

def run_demo_query(query_data):
    """Run a single demo query"""
    query = query_data["query"]
    description = query_data["description"]
    
    print(f"\n{'='*60}")
    print(f"🔍 Demo Query: {query}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    try:
        # Send query to API
        response = requests.post(
            f"{BASE_URL}/api/query",
            json={"query": query, "k": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("error"):
                print(f"❌ Error: {data['error']}")
                return False
            
            # Display results
            print(f"\n📊 Results Summary:")
            print(f"   • Query: {data.get('query', 'N/A')}")
            print(f"   • Results found: {data.get('results_count', 0)}")
            print(f"   • Images found: {data.get('images_count', 0)}")
            
            print(f"\n🤖 AI Response:")
            print("-" * 40)
            response_text = data.get('response', 'No response')
            # Truncate very long responses for demo
            if len(response_text) > 500:
                response_text = response_text[:497] + "..."
            print(response_text)
            
            # Display image information
            images = data.get('images', [])
            if images:
                print(f"\n🖼️  Related Images ({len(images)}):")
                print("-" * 40)
                
                for i, img in enumerate(images[:5], 1):  # Show first 5 images
                    status = "✅" if img.get('exists', False) else "❌"
                    print(f"{i}. {img.get('filename', 'Unknown')} {status}")
                    print(f"   Page: {img.get('page', 'N/A')}")
                    print(f"   Path: {img.get('web_path', 'N/A')}")
                    
                    if img.get('ocr_text'):
                        ocr_preview = img['ocr_text'][:100] + "..." if len(img['ocr_text']) > 100 else img['ocr_text']
                        print(f"   Content: {ocr_preview}")
                    print()
                
                if len(images) > 5:
                    print(f"   ... and {len(images) - 5} more images")
            else:
                print(f"\n🖼️  No images found for this query")
            
            print(f"\n✅ Demo query completed successfully!")
            return True
            
        else:
            print(f"❌ API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Query timed out (30 seconds)")
        print("   This might happen on first query due to model loading")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def run_interactive_demo():
    """Run interactive demo mode"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE DEMO MODE")
    print("="*60)
    print("Enter your own queries to test the system!")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print()
    
    while True:
        try:
            query = input("🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', '']:
                break
            
            demo_data = {
                "query": query,
                "description": "User-provided query"
            }
            
            run_demo_query(demo_data)
            
        except KeyboardInterrupt:
            print("\n👋 Interactive demo stopped!")
            break
        except EOFError:
            break

def main():
    """Main demo function"""
    print("🎬 FRC RAG Chat Interface Demo")
    print("="*60)
    print("This demo will show you the capabilities of the web interface")
    print("by running example queries and displaying the results.")
    print()
    
    # Check if server is running
    if not check_server():
        print("\n💡 To start the server, run:")
        print("   python app.py")
        print("   or")
        print("   ./start_web.sh")
        return
    
    print(f"\n🚀 Running {len(DEMO_QUERIES)} demo queries...")
    
    # Run demo queries
    successful_queries = 0
    for i, query_data in enumerate(DEMO_QUERIES, 1):
        print(f"\n[{i}/{len(DEMO_QUERIES)}] Running demo query...")
        
        if run_demo_query(query_data):
            successful_queries += 1
        
        # Pause between queries for readability
        if i < len(DEMO_QUERIES):
            time.sleep(2)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📊 DEMO SUMMARY")
    print(f"="*60)
    print(f"✅ Successful queries: {successful_queries}/{len(DEMO_QUERIES)}")
    print(f"🌐 Web interface: {BASE_URL}")
    print(f"📱 Open the URL above in your browser for the full chat experience!")
    
    if successful_queries < len(DEMO_QUERIES):
        print(f"\n⚠️  Some queries failed. This might be due to:")
        print("   • Database not properly initialized")
        print("   • Ollama service not running")
        print("   • Missing PDF data or images")
    
    # Offer interactive mode
    print(f"\n💬 Want to try your own queries?")
    response = input("Start interactive demo? (y/N): ").lower()
    
    if response in ['y', 'yes']:
        run_interactive_demo()
    
    print(f"\n👋 Demo completed! Enjoy using the FRC RAG Chat Interface!")

if __name__ == "__main__":
    main()
