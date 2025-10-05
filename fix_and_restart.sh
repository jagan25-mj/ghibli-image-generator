#!/bin/bash
# Complete fix and restart script

echo "🔧 Fixing CSS issues and restarting servers..."
echo ""

# Stop all servers
echo "🛑 Stopping all running servers..."
pkill -f "manage.py runserver" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null
pkill -f "vite" 2>/dev/null
tmux kill-session -t django 2>/dev/null
tmux kill-session -t react 2>/dev/null

sleep 1

# Clear caches
echo "🧹 Clearing caches..."
cd /workspace/frontend
rm -rf node_modules/.vite dist .vite 2>/dev/null
cd /workspace

echo "✅ Cleanup complete"
echo ""

# Install tmux if needed
if ! command -v tmux &> /dev/null; then
    echo "📦 Installing tmux..."
    sudo apt-get update -qq && sudo apt-get install -y -qq tmux
fi

# Start Django
echo "🐍 Starting Django backend..."
tmux new-session -d -s django 'cd /workspace && python manage.py runserver 0.0.0.0:8000'

sleep 2

# Start React
echo "⚛️  Starting React frontend..."
tmux new-session -d -s react 'cd /workspace/frontend && npm run dev -- --host 0.0.0.0'

echo ""
echo "=================================================="
echo "✅ Servers restarted with fixes applied!"
echo ""
echo "📱 Access your app:"
echo "   1. Click 'PORTS' tab at bottom of VS Code"
echo "   2. Find port 5173"
echo "   3. Click the 🌐 globe icon"
echo ""
echo "🔍 Check server logs:"
echo "   Django: tmux attach -t django"
echo "   React:  tmux attach -t react"
echo "   (Ctrl+B then D to detach)"
echo ""
echo "The CSS errors should now be gone! 🎉"
echo "=================================================="