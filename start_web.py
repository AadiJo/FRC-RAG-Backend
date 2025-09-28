#!/usr/bin/env python3
"""
FRC RAG Chat Interface Launcher
Cross-platform Python launcher for the web interface
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python():
    """Check if Python is available"""
    print("🐍 Checking Python version...")
    print(f"   Python {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    return True

def setup_venv():
    """Set up virtual environment if needed"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Get the correct python executable path
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return str(python_exe), str(pip_exe)

def install_requirements(pip_exe):
    """Install required packages"""
    print("📚 Installing/updating dependencies...")
    try:
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"   Error output: {e.stderr}")
        return False
    return True

def check_database():
    """Check if database exists"""
    db_paths = ["chroma_enhanced", "db"]
    
    for db_path in db_paths:
        if Path(db_path).exists():
            print(f"✅ Database found: {db_path}")
            return True
    
    print("⚠️  No database found!")
    print("   You need to create the database first:")
    print("   Run: python create_database.py")
    print("   or:  python create_database2.py (for enhanced version)")
    
    response = input("\nDo you want to continue anyway? (y/N): ").lower()
    return response in ['y', 'yes']

def check_ollama():
    """Check if Ollama is available (will be started automatically)"""
    print("🤖 Ollama will be started automatically when the application launches")
    
    try:
        # Just check if ollama command exists
        subprocess.run(['ollama', '--version'], 
                      capture_output=True, 
                      check=True, 
                      timeout=5)
        print("✅ Ollama is installed and ready")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Ollama command not working properly")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama from https://ollama.ai/")
        print("   The application will still try to start, but AI features may not work")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama command timed out")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Ollama: {e}")
        return False

def start_web_server(python_exe):
    """Start the Flask web server"""
    print("\n" + "="*50)
    print("✅ Setup complete! Starting the web interface...")
    print("📱 Open your browser to: http://localhost:5000")
    print("💬 Chat interface will be ready in a few seconds...")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    
    try:
        # Start the Flask application
        subprocess.run([python_exe, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Starting FRC RAG Chat Interface...")
    print("="*50)
    
    # Check Python version
    if not check_python():
        sys.exit(1)
    
    # Set up virtual environment
    try:
        python_exe, pip_exe = setup_venv()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements(pip_exe):
        sys.exit(1)
    
    # Check database
    if not check_database():
        sys.exit(1)
    
    # Check Ollama (but continue regardless since it will be started automatically)
    check_ollama()
    
    # Start web server
    if not start_web_server(python_exe):
        sys.exit(1)

if __name__ == "__main__":
    main()
