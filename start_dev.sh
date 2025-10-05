#!/bin/bash

# Development startup script for Ghibli Image Generator
# This script starts both Django backend and React frontend

set -e

echo "🎨 Starting Ghibli Image Generator Development Environment"
echo "=========================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $DJANGO_PID $REACT_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Django backend
echo "🐍 Starting Django backend..."
python3 manage.py migrate --noinput
python3 manage.py runserver 8000 &
DJANGO_PID=$!
echo "✅ Django backend started (PID: $DJANGO_PID) at http://localhost:8000"
echo ""

# Wait a moment for Django to start
sleep 2

# Start React frontend
echo "⚛️  Starting React frontend..."
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
REACT_PID=$!
echo "✅ React frontend started (PID: $REACT_PID) at http://localhost:5173"
echo ""

cd ..

echo "=========================================================="
echo "🚀 Both servers are running!"
echo ""
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API:      http://localhost:8000/api/"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=========================================================="

# Wait for processes
wait $DJANGO_PID $REACT_PID