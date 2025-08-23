#!/bin/bash

# ML-Powered Code Review Agent - Server Startup Script

echo "🚀 Starting ML-Powered Code Review Agent..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Please create a virtual environment first:"
    echo "   python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "✅ Starting server on http://127.0.0.1:8000"
echo "📚 API Documentation: http://127.0.0.1:8000/docs"
echo "🔍 Health Check: http://127.0.0.1:8000/health"
echo "=================================================="
echo "Press Ctrl+C to stop the server"
echo ""

python3 start_server.py
