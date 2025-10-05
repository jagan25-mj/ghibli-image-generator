# 🚀 Run in GitHub Codespaces - Simple Guide

## ⚡ Super Quick Start

Copy and paste these commands in your Codespace terminal:

```bash
# 1. Run the automated setup script
./start_codespaces.sh
```

**That's literally it!** 🎉

The script handles everything:
- Installs Python dependencies
- Installs Node.js dependencies  
- Sets up the database
- Starts both servers

## 📱 Access Your App

After running the script:

1. Look at the bottom of VS Code
2. Click the **"PORTS"** tab
3. Find **port 5173** (React Frontend)
4. Click the **🌐 globe icon** next to it
5. Your app opens in a new browser tab!

## 🛑 Stop Everything

```bash
./stop_codespaces.sh
```

---

## 📝 Manual Step-by-Step (If You Want Control)

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

### Step 4: Start Django Server
Open a terminal and run:
```bash
python manage.py runserver 0.0.0.0:8000
```
**Keep this terminal open!**

### Step 5: Start React Server
Open a **NEW** terminal (click the + button) and run:
```bash
cd frontend
npm run dev -- --host 0.0.0.0
```
**Keep this terminal open too!**

### Step 6: Access Your App
Go to the **PORTS** tab and click the globe icon next to port **5173**.

---

## 🔍 Check If It's Working

### Test Django API:
```bash
curl http://localhost:8000/api/config/
```
Should return JSON with configuration options.

### Test React:
```bash
curl http://localhost:5173
```
Should return HTML content.

### Check Running Processes:
```bash
ps aux | grep -E "manage.py|npm"
```
Should show both Django and Node processes.

---

## 💡 Important Notes for Codespaces

### Always Bind to 0.0.0.0
❌ **DON'T USE:** `python manage.py runserver` (binds to 127.0.0.1)
✅ **USE:** `python manage.py runserver 0.0.0.0:8000`

❌ **DON'T USE:** `npm run dev`
✅ **USE:** `npm run dev -- --host 0.0.0.0`

**Why?** Codespaces needs 0.0.0.0 for port forwarding to work!

### Use the PORTS Tab
- Don't try to access `localhost` directly
- Use the forwarded URLs from the PORTS tab
- The URLs look like: `https://xxxxx-5173.app.github.dev`

### Making Ports Public
If you want to share your app with someone:
1. Right-click the port in PORTS tab
2. Select **"Port Visibility"**
3. Choose **"Public"**

---

## 🎯 Quick Commands Reference

| Command | What It Does |
|---------|--------------|
| `./start_codespaces.sh` | Start everything automatically |
| `./stop_codespaces.sh` | Stop all servers |
| `tmux attach -t django` | View Django server logs |
| `tmux attach -t react` | View React server logs |
| `Ctrl+B then D` | Exit tmux (keeps server running) |
| `pip install -r requirements.txt` | Install Python packages |
| `cd frontend && npm install` | Install Node packages |
| `python manage.py migrate` | Setup database |

---

## 🐛 Troubleshooting

### "Address already in use" Error
```bash
./stop_codespaces.sh
./start_codespaces.sh
```

### "Module not found" Error
```bash
# For Python
pip install -r requirements.txt --force-reinstall

# For Node
cd frontend
rm -rf node_modules package-lock.json
npm install
cd ..
```

### Can't See PORTS Tab
Look at the **bottom panel** of VS Code. You should see tabs for:
- PROBLEMS
- OUTPUT
- DEBUG CONSOLE
- TERMINAL
- **PORTS** ← Click this one!

### Codespace Timeout
If your Codespace times out and stops:
1. Just reopen it from GitHub
2. Run `./start_codespaces.sh` again
3. Everything will restart

---

## 🎨 Generate Your First Image

1. Open the app from the PORTS tab (port 5173)
2. You'll see a beautiful dark-themed interface
3. Type a prompt like: **"A cozy village at dusk with lanterns and soft lighting"**
4. Click **"Generate"**
5. Wait about 15-20 seconds
6. Your AI-generated Ghibli-style image appears!
7. Click **"Download"** to save it

---

## 📊 What You Should See

### After Running `./start_codespaces.sh`:
```
🎨 Starting Ghibli Image Generator in Codespaces
==================================================

📦 Installing Python dependencies...
🗄️  Setting up database...
📦 Installing Node.js dependencies...
🐍 Starting Django backend on port 8000...
⚛️  Starting React frontend on port 5173...

==================================================
✅ Both servers are running!

📱 Access your application:
   1. Click the 'PORTS' tab in VS Code (bottom panel)
   2. Find port 5173 (React Frontend)
   3. Click the 🌐 globe icon to open in browser
==================================================
```

### In the PORTS Tab:
```
Port    Process              Visibility
8000    Django (Python)      Private
5173    React (Vite)         Private
```

Click the 🌐 icon next to **5173** to open your app!

---

## 📚 More Documentation

- **Quick Setup**: [CODESPACES_QUICKSTART.md](CODESPACES_QUICKSTART.md)
- **Detailed Guide**: [CODESPACES_SETUP.md](CODESPACES_SETUP.md)
- **General Docs**: [README.md](README.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## ✅ Checklist

- [ ] Codespace is open in VS Code
- [ ] Ran `./start_codespaces.sh` successfully
- [ ] Both servers show as running in logs
- [ ] PORTS tab shows ports 8000 and 5173
- [ ] Clicked globe icon on port 5173
- [ ] App opens in browser
- [ ] Can type in the prompt field
- [ ] Can click Generate button
- [ ] Image generates successfully

If all checked ✅, you're ready to create beautiful AI art! 🎨

---

**Happy Generating in the Cloud! ☁️✨**