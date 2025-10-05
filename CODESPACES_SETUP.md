# 🚀 GitHub Codespaces Setup Guide

Complete guide to run the Ghibli Image Generator in GitHub Codespaces.

## Quick Start (Copy & Paste)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Setup Database
```bash
python manage.py migrate
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Start Both Servers

**Option A: Using tmux (Recommended for Codespaces)**
```bash
# Install tmux if not available
sudo apt-get update && sudo apt-get install -y tmux

# Start both servers in tmux
tmux new-session -d -s django 'python manage.py runserver 0.0.0.0:8000'
tmux new-session -d -s react 'cd frontend && npm run dev -- --host 0.0.0.0'

# View server logs (optional)
tmux attach -t django    # Press Ctrl+B then D to detach
tmux attach -t react     # Press Ctrl+B then D to detach

echo "✅ Both servers are running!"
echo "Django: http://localhost:8000"
echo "React: http://localhost:5173"
```

**Option B: Using Background Processes**
```bash
# Start Django in background
nohup python manage.py runserver 0.0.0.0:8000 > django.log 2>&1 &
echo $! > django.pid

# Start React in background
cd frontend
nohup npm run dev -- --host 0.0.0.0 > ../react.log 2>&1 &
echo $! > ../react.pid
cd ..

echo "✅ Both servers are running!"
echo "Check logs: tail -f django.log react.log"
```

**Option C: Two Terminal Windows (Manual)**
```bash
# Terminal 1: Django
python manage.py runserver 0.0.0.0:8000

# Terminal 2: React (open new terminal)
cd frontend
npm run dev -- --host 0.0.0.0
```

### Step 5: Access the Application

GitHub Codespaces will automatically detect the ports and show you forwarded URLs.

1. **Click on the "PORTS" tab** in VS Code bottom panel
2. **Find ports 5173 (React) and 8000 (Django)**
3. **Click the globe icon** to open in browser
4. **Or copy the forwarded URL** (looks like: `https://xxxxx-5173.app.github.dev`)

## Port Forwarding in Codespaces

Codespaces automatically forwards ports, but you can also do it manually:

```bash
# View forwarded ports
gh codespace ports

# Forward port manually (if needed)
gh codespace ports forward 5173:5173
gh codespace ports forward 8000:8000
```

## Environment Configuration for Codespaces

Create `.env` file in frontend directory:
```bash
cd frontend
cat > .env << 'EOF'
VITE_API_URL=http://localhost:8000
EOF
cd ..
```

## Stopping the Servers

**If using tmux:**
```bash
tmux kill-session -t django
tmux kill-session -t react
```

**If using background processes:**
```bash
# Kill servers using saved PIDs
kill $(cat django.pid) 2>/dev/null
kill $(cat react.pid) 2>/dev/null

# Or find and kill processes
pkill -f "python manage.py runserver"
pkill -f "npm run dev"
```

## Checking Server Status

```bash
# Check if servers are running
ps aux | grep "manage.py runserver"
ps aux | grep "npm run dev"

# Check logs
tail -f django.log
tail -f react.log
```

## Troubleshooting in Codespaces

### Port Already in Use
```bash
# Kill processes on specific ports
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

### Module Not Found Errors
```bash
# Reinstall Python packages
pip install -r requirements.txt --upgrade --force-reinstall

# Reinstall Node packages
cd frontend
rm -rf node_modules package-lock.json
npm install
cd ..
```

### Permission Errors
```bash
# Make scripts executable
chmod +x start_dev.sh
```

### Out of Memory
```bash
# Check memory usage
free -h

# Clear npm cache
npm cache clean --force
```

## Complete One-Command Setup

Save this as `codespaces_start.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up Ghibli Image Generator in Codespaces..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Setup database
echo "🗄️  Setting up database..."
python manage.py migrate --noinput

# Install frontend dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install --silent
cd ..

# Start servers with tmux
echo "🚀 Starting servers..."
sudo apt-get update -qq && sudo apt-get install -y -qq tmux

tmux new-session -d -s django 'python manage.py runserver 0.0.0.0:8000'
tmux new-session -d -s react 'cd frontend && npm run dev -- --host 0.0.0.0'

echo ""
echo "✅ Setup complete!"
echo ""
echo "📱 Access your app:"
echo "   1. Click 'PORTS' tab below"
echo "   2. Find port 5173 (React Frontend)"
echo "   3. Click the globe icon to open"
echo ""
echo "🔧 Useful commands:"
echo "   View Django logs:  tmux attach -t django"
echo "   View React logs:   tmux attach -t react"
echo "   Stop servers:      tmux kill-session -t django && tmux kill-session -t react"
echo ""
echo "Press Ctrl+B then D to detach from tmux"
```

Then run:
```bash
chmod +x codespaces_start.sh
./codespaces_start.sh
```

## VS Code Tasks (Optional)

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Django",
      "type": "shell",
      "command": "python manage.py runserver 0.0.0.0:8000",
      "problemMatcher": [],
      "isBackground": true
    },
    {
      "label": "Start React",
      "type": "shell",
      "command": "cd frontend && npm run dev -- --host 0.0.0.0",
      "problemMatcher": [],
      "isBackground": true
    },
    {
      "label": "Start All Servers",
      "dependsOn": ["Start Django", "Start React"],
      "problemMatcher": []
    }
  ]
}
```

Then use: `Ctrl+Shift+P` → `Tasks: Run Task` → `Start All Servers`

## Dev Container Configuration (Optional)

Create `.devcontainer/devcontainer.json` for automatic setup:

```json
{
  "name": "Ghibli Generator",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {
      "version": "18"
    }
  },
  "forwardPorts": [8000, 5173],
  "postCreateCommand": "pip install -r requirements.txt && python manage.py migrate && cd frontend && npm install",
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install Python deps |
| `python manage.py migrate` | Setup database |
| `cd frontend && npm install` | Install Node deps |
| `python manage.py runserver 0.0.0.0:8000` | Start Django |
| `cd frontend && npm run dev -- --host 0.0.0.0` | Start React |
| `tmux attach -t django` | View Django logs |
| `tmux attach -t react` | View React logs |
| `Ctrl+B then D` | Detach from tmux |

## Tips for Codespaces

1. **Use tmux** for persistent sessions
2. **Bind to 0.0.0.0** not localhost (important for port forwarding)
3. **Check PORTS tab** for forwarded URLs
4. **Make port public** if you need to share (right-click port → Port Visibility → Public)
5. **Save your work** - Codespace may timeout after inactivity

## Testing the Setup

```bash
# Test Django API
curl http://localhost:8000/api/config/

# Should return JSON with configuration

# Test React (from within Codespace)
curl http://localhost:5173

# Should return HTML
```

## Memory Considerations

Codespaces have limited memory. If you encounter issues:

```bash
# Use speed preset (faster, less memory)
# Reduce inference steps in UI
# Close other applications
# Upgrade Codespace machine type if available
```

## Persistence

To keep your setup between sessions:

1. **Commit your changes** to git
2. **Extensions persist** automatically
3. **Forward ports persist** in the codespace
4. **Restart servers** after codespace resumes

---

**Ready to generate beautiful images in the cloud! 🎨☁️**