# 🚀 Quick Start Guide

Get the Ghibli Image Generator running in 5 minutes!

## Prerequisites Check

```bash
# Check Python (need 3.9+)
python3 --version

# Check Node.js (need 18+)
node --version

# Check pip
pip3 --version

# Check npm
npm --version
```

If any are missing, install them first:
- **Python**: https://www.python.org/downloads/
- **Node.js**: https://nodejs.org/

## Installation (5 steps)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs Django, Stable Diffusion, and all backend dependencies.

### 2. Set Up Database

```bash
python manage.py migrate
```

Creates the SQLite database.

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

Installs React and all frontend dependencies.

### 4. Start the Application

**Automated (Easiest):**
```bash
# Linux/Mac
./start_dev.sh

# Windows
start_dev.bat
```

**Manual:**
```bash
# Terminal 1: Start Django
python manage.py runserver

# Terminal 2: Start React (in new terminal)
cd frontend
npm run dev
```

### 5. Open Your Browser

Visit: **http://localhost:5173**

That's it! 🎉

## First Image Generation

1. Enter a prompt: `"A cozy village at dusk with lanterns"`
2. Click **Generate**
3. Wait 10-20 seconds
4. Download your Ghibli-style image!

## Troubleshooting

### Port Already in Use

**Django (8000):**
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**React (5173):**
```bash
# Linux/Mac
lsof -ti:5173 | xargs kill -9

# Windows
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

### Module Not Found

**Python:**
```bash
pip install -r requirements.txt --upgrade
```

**Node:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS Errors

Make sure both servers are running and React is at `http://localhost:5173`.

### GPU Not Available

The app will use CPU mode automatically (slower but works).

## Quick Tips

### Best Prompts

Good prompts are specific:
- ✅ "A peaceful valley with a small cottage, cherry blossoms, golden hour lighting"
- ❌ "A nice picture"

### Speed vs Quality

- **Speed Preset**: ~10 seconds, good quality
- **Balanced Preset**: ~15 seconds, great quality
- **Quality Preset**: ~25 seconds, best quality

### Using Negative Prompts

Add things to avoid:
```
Negative: "low quality, blurry, watermark, text, signature"
```

### Reproducible Results

Copy the seed from a generated image to recreate similar images.

## Next Steps

- Read `README.md` for full documentation
- See `DEPLOYMENT.md` for production deployment
- Check `PERFORMANCE_OPTIMIZATIONS.md` for technical details

## Common Commands

```bash
# Start development
./start_dev.sh

# Build React for production
cd frontend && npm run build

# Run Django tests
python manage.py test

# Check Django admin
python manage.py createsuperuser
# Visit: http://localhost:8000/admin

# Lint frontend code
cd frontend && npm run lint
```

## Getting Help

1. Check the error message carefully
2. Look in browser console (F12)
3. Check Django terminal for backend errors
4. Check Vite terminal for frontend errors
5. Read the documentation files

## Success Checklist

- [ ] Python 3.9+ installed
- [ ] Node.js 18+ installed
- [ ] Dependencies installed (pip + npm)
- [ ] Database migrated
- [ ] Both servers running
- [ ] Browser shows the app
- [ ] Can generate an image
- [ ] No errors in console

If all checked, you're good to go! 🎨

---

**Need More Help?** Check the other documentation files or open an issue.