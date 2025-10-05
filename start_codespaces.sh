#!/bin/bash
# GitHub Codespaces startup script for Ghibli Image Generator

set -e

echo "🎨 Starting Ghibli Image Generator in Codespaces"
echo "=================================================="
echo ""

# Check if running in Codespaces
if [ -z "$CODESPACES" ]; then
    echo "⚠️  Warning: This script is optimized for GitHub Codespaces"
    echo "   For local development, use ./start_dev.sh instead"
    echo ""
fi

# Install Python dependencies if needed
if ! python -c "import django" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -q -r requirements.txt
fi

# Run migrations if needed
if [ ! -f "db.sqlite3" ]; then
    echo "🗄️  Setting up database..."
    python manage.py migrate --noinput
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd frontend
    npm install --silent
    cd ..
fi

# Install tmux if not available
if ! command -v tmux &> /dev/null; then
    echo "📦 Installing tmux..."
    sudo apt-get update -qq && sudo apt-get install -y -qq tmux
fi

# Kill any existing sessions
tmux kill-session -t django 2>/dev/null || true
tmux kill-session -t react 2>/dev/null || true

# Start Django server
echo "🐍 Starting Django backend on port 8000..."
tmux new-session -d -s django 'python manage.py runserver 0.0.0.0:8000'

# Wait a moment for Django to start
sleep 2

# Start React dev server
echo "⚛️  Starting React frontend on port 5173..."
tmux new-session -d -s react 'cd frontend && npm run dev -- --host 0.0.0.0'

echo ""
echo "=================================================="
echo "✅ Both servers are running!"
echo ""
echo "📱 Access your application:"
echo "   1. Click the 'PORTS' tab in VS Code (bottom panel)"
echo "   2. Find port 5173 (React Frontend)"
echo "   3. Click the 🌐 globe icon to open in browser"
echo ""
echo "🔍 View server logs:"
echo "   Django:  tmux attach -t django"
echo "   React:   tmux attach -t react"
echo "   (Press Ctrl+B then D to detach)"
echo ""
echo "🛑 Stop servers:"
echo "   ./stop_codespaces.sh"
echo "   OR"
echo "   tmux kill-session -t django && tmux kill-session -t react"
echo ""
echo "=================================================="