# 🔄 How to Restart Servers After Fix

## Stop All Running Servers

```bash
# Kill any existing processes
pkill -f "manage.py runserver" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null
pkill -f "vite" 2>/dev/null

# Or if using tmux
tmux kill-session -t django 2>/dev/null
tmux kill-session -t react 2>/dev/null
```

## Clear Caches

```bash
cd /workspace/frontend
rm -rf node_modules/.vite dist
cd /workspace
```

## Start Fresh

### Option 1: Automated (Recommended)
```bash
cd /workspace
./start_codespaces.sh
```

### Option 2: Manual

**Terminal 1 - Django:**
```bash
cd /workspace
python manage.py runserver 0.0.0.0:8000
```

**Terminal 2 - React:**
```bash
cd /workspace/frontend
npm run dev -- --host 0.0.0.0
```

## Verify It's Working

1. **Check for errors in terminal** - should be clean
2. **Open PORTS tab** in VS Code
3. **Click globe icon** next to port 5173
4. **App should load** without overlay errors

## If Still Having Issues

### Full Reset
```bash
cd /workspace

# Stop everything
pkill -f "python" 2>/dev/null
pkill -f "node" 2>/dev/null

# Clear all caches
rm -rf frontend/node_modules/.vite
rm -rf frontend/dist

# Reinstall frontend deps (if needed)
cd frontend
npm install
cd ..

# Restart
./start_codespaces.sh
```

### Check for Syntax Errors
```bash
# Verify CSS syntax
cd /workspace/frontend
npx tailwindcss -i ./src/index.css -o /dev/null

# Should not show errors
```

## Expected Output

When you start the React dev server, you should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://0.0.0.0:5173/
  ➜  press h + enter to show help
```

No errors about `border-border` or any CSS issues!