#!/bin/bash
# Stop servers in GitHub Codespaces

echo "🛑 Stopping Ghibli Image Generator servers..."

# Kill tmux sessions
tmux kill-session -t django 2>/dev/null && echo "✅ Django server stopped" || echo "⚠️  Django server not running"
tmux kill-session -t react 2>/dev/null && echo "✅ React server stopped" || echo "⚠️  React server not running"

# Also kill any remaining processes on the ports
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true

echo ""
echo "✅ All servers stopped"