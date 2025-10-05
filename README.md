# 🎨 Ghibli Image Generator

A beautiful, high-performance web application for generating Ghibli-style images using Stable Diffusion, featuring a modern React.js frontend and Django REST API backend.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18.2-blue.svg)

## ✨ Features

### 🎯 Core Functionality
- **Text-to-Image Generation**: Create stunning Ghibli-style images from text prompts
- **Image-to-Image Transformation**: Transform existing images with Ghibli aesthetics
- **Multiple Presets**: Balanced, Speed, Quality, Faithful, and Stylized modes
- **Advanced Controls**: Fine-tune guidance, steps, strength, and more
- **Upscaling**: Built-in image upscaling with Real-ESRGAN or Lanczos
- **LoRA Support**: Use custom LoRA models for specialized styles

### 🚀 Performance Optimizations
- **Code Splitting**: Lazy-loaded React components for minimal initial bundle
- **React Query**: Efficient data fetching with automatic caching and deduplication
- **Image Lazy Loading**: Images load only when visible
- **Memoization**: Optimized re-renders with React.memo and useCallback
- **Error Boundaries**: Graceful error handling prevents app crashes
- **Progressive Web App**: Fast, responsive, and installable

### 💅 UI/UX Excellence
- **Modern Design**: Beautiful glass-morphism interface with smooth animations
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile
- **Real-time Progress**: Live progress indicators during generation
- **Toast Notifications**: Non-intrusive feedback for user actions
- **Advanced Settings**: Collapsible advanced options for power users
- **Accessibility**: WCAG compliant with keyboard navigation and ARIA labels

### 🔧 Technical Stack

**Frontend:**
- ⚛️ React 18 with hooks
- ⚡️ Vite for lightning-fast builds
- 🎨 Tailwind CSS for styling
- 🎭 Framer Motion for animations
- 📝 React Hook Form for form management
- 🔄 TanStack Query for data fetching
- 🍞 React Hot Toast for notifications

**Backend:**
- 🐍 Django 4.2+ REST API
- 🖼️ Stable Diffusion 1.5
- 🎨 Custom pipeline for Ghibli style
- 🔒 CORS support for frontend integration
- 📦 Optimized image processing

## 📦 Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (recommended for faster generation)

### Quick Start

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ghibli-generator
```

2. **Set up the backend**:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start Django server
python manage.py runserver
```

3. **Set up the frontend** (in a new terminal):
```bash
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```

4. **Open your browser**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## 🎯 Usage

### Basic Generation

1. Enter a descriptive prompt (e.g., "A cozy village at dusk with lanterns")
2. Select a preset (Balanced, Speed, or Quality)
3. Click "Generate" and wait for your image
4. Download or open the generated image

### Advanced Options

Click "Advanced Settings" to access:
- **Strength**: Control how much the init image influences the output (img2img)
- **Guidance Scale**: How closely to follow the prompt (1-20)
- **Inference Steps**: Quality vs. speed trade-off (6-60)
- **Aspect Ratio**: Choose from Square, Landscape, Wide, or Portrait
- **Seed**: Use a specific seed for reproducible results
- **Negative Prompt**: Specify what to avoid in the image
- **LoRA**: Load custom LoRA models
- **Upscaling**: Choose upscale mode and factor

### Tips for Best Results

- **Detailed Prompts**: More details = better results
- **Negative Prompts**: Use to avoid unwanted elements (e.g., "low quality, blurry, text")
- **Preset Selection**: 
  - Speed: Quick results (7s guidance, 14 steps)
  - Balanced: Good quality/speed balance (7.5s guidance, 18 steps)
  - Quality: Best quality (8s guidance, 24 steps)
- **Seed Reuse**: Copy the seed from metadata to recreate similar images

## 📊 Performance Benchmarks

### Frontend Performance
- **Initial Load**: ~150KB gzipped (with code splitting)
- **Time to Interactive**: <2s on 3G
- **Lighthouse Score**: 95+ (Performance, Accessibility, Best Practices)

### Backend Performance
- **Generation Time** (on GPU):
  - Speed preset: ~8-12s
  - Balanced preset: ~12-18s
  - Quality preset: ~20-30s
- **API Response**: <100ms for config endpoint

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐
│   React     │   API   │   Django     │
│  Frontend   │◄───────►│   Backend    │
│  (Vite)     │  HTTP   │  (REST API)  │
└─────────────┘         └──────────────┘
      │                        │
      │                        ▼
      │                  ┌──────────────┐
      │                  │   Stable     │
      │                  │  Diffusion   │
      │                  │   Pipeline   │
      └──────────────────┤              │
           Display       └──────────────┘
```

## 🔐 Security

- CSRF protection enabled
- CORS properly configured
- Input validation on both frontend and backend
- File upload size limits
- Secure headers configured
- No sensitive data in client-side code

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions, including:
- Production setup
- Environment configuration
- Scaling strategies
- Performance tuning
- Security checklist

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is not affiliated with Studio Ghibli. All generated images are created locally using AI models and are not official Studio Ghibli content.

## 🙏 Acknowledgments

- Stable Diffusion by Stability AI
- React and the React ecosystem
- Django framework
- Tailwind CSS
- All open-source contributors

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

Made with ❤️ for the love of beautiful AI-generated art