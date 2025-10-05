# ⚡ GitHub Codespaces - Quick Start

## 🚀 One-Command Start (Recommended)

```bash
./start_codespaces.sh
```

That's it! The script will:
- ✅ Install all dependencies automatically
- ✅ Setup the database
- ✅ Start both Django and React servers
- ✅ Show you how to access the app

## 📱 Access Your App

1. **Click the "PORTS" tab** in VS Code (bottom panel)
2. **Find port 5173** (React Frontend)
3. **Click the 🌐 globe icon** to open in browser

Your app will open in a new tab!

## 🛑 Stop Servers

```bash
./stop_codespaces.sh
```

## 📋 Manual Commands (If You Prefer)

### Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Setup Database
```bash
python manage.py migrate
```

### Start Servers

**Terminal 1: Django**
```bash
python manage.py runserver 0.0.0.0:8000
```

**Terminal 2: React** (open new terminal)
```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

## 🔍 View Server Logs

```bash
# View Django logs
tmux attach -t django

# View React logs
tmux attach -t react

# Press Ctrl+B then D to detach
```

## ⚡ Quick Tips

- **Bind to 0.0.0.0** (not localhost) for port forwarding to work
- **Use the PORTS tab** to access your app
- **Make port public** if you want to share (right-click port → Port Visibility → Public)
- **Servers persist** in tmux sessions even if you close terminals

## 🔧 Troubleshooting

### Port Already in Use
```bash
./stop_codespaces.sh
./start_codespaces.sh
```

### Module Not Found
```bash
pip install -r requirements.txt --force-reinstall
cd frontend && rm -rf node_modules && npm install
```

### See All Processes
```bash
ps aux | grep -E "manage.py|npm"
```

## 📊 Test Your Setup

```bash
# Test Django API
curl http://localhost:8000/api/config/

# Test React Frontend  
curl http://localhost:5173
```

Both should return responses without errors.

## 🎨 Generate Your First Image

1. Open the app from PORTS tab
2. Enter prompt: **"A peaceful village at sunset with lanterns"**
3. Click **"Generate"**
4. Wait ~15-20 seconds
5. Your beautiful Ghibli-style image appears!

---

**Need more help?** See [CODESPACES_SETUP.md](CODESPACES_SETUP.md) for detailed instructions.