#!/bin/bash
# Eburon TTS Deployment Script

set -e

echo "🚀 Eburon TTS Deployment"
echo "========================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found${NC}"

# Install dependencies if not present
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""
echo "Starting Eburon TTS server..."
echo "========================"
echo ""
echo "🌐 Web UI: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""

python eburon_tts_server.py
