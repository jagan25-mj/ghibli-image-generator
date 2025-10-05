# Deployment Guide

This guide explains how to deploy the Ghibli Image Generator with both Django backend and React frontend.

## Architecture

The application consists of:
- **Backend**: Django REST API serving image generation endpoints
- **Frontend**: React SPA built with Vite for optimal performance

## Development Setup

### Backend (Django)

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run migrations**:
```bash
python manage.py migrate
```

3. **Start Django development server**:
```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`

### Frontend (React)

1. **Install Node.js dependencies**:
```bash
cd frontend
npm install
```

2. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env if needed to point to a different backend URL
```

3. **Start Vite development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Production Deployment

### Option 1: Separate Deployment (Recommended)

Deploy frontend and backend separately for better scalability.

#### Backend Deployment

1. **Set environment variables**:
```bash
export DJANGO_DEBUG=0
export DJANGO_SECRET_KEY=<your-secret-key>
export DJANGO_ALLOWED_HOSTS=your-domain.com
export CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com
```

2. **Collect static files**:
```bash
python manage.py collectstatic --noinput
```

3. **Run with Gunicorn** (or your preferred WSGI server):
```bash
pip install gunicorn
gunicorn ghibli_site.wsgi:application --bind 0.0.0.0:8000
```

#### Frontend Deployment

1. **Build for production**:
```bash
cd frontend
npm run build
```

2. **Deploy the `dist/` directory** to your hosting service:
   - Netlify
   - Vercel
   - AWS S3 + CloudFront
   - Any static hosting service

3. **Configure environment variables** on your hosting platform:
   - `VITE_API_URL`: URL of your backend API

### Option 2: Integrated Deployment

Serve the React build from Django.

1. **Build React frontend**:
```bash
cd frontend
npm run build
```

2. **Configure Django settings**:

Add to `ghibli_site/settings.py`:
```python
# Serve React build
REACT_BUILD_DIR = BASE_DIR / 'frontend' / 'dist'
STATICFILES_DIRS = [BASE_DIR / "static", REACT_BUILD_DIR / 'assets']
```

3. **Update Django URLs** to serve React:

Add to `ghibli_site/urls.py`:
```python
from django.views.generic import TemplateView
from django.conf import settings

urlpatterns = [
    # ... existing patterns ...
    path('', TemplateView.as_view(
        template_name='index.html',
        extra_context={'settings': settings}
    )),
]

# Configure template directory
TEMPLATES[0]['DIRS'] = [REACT_BUILD_DIR]
```

4. **Deploy with your preferred method**

## Performance Considerations

### Backend Optimizations

- Use a production-ready WSGI server (Gunicorn, uWSGI)
- Enable caching for static assets
- Use a reverse proxy (Nginx) for serving media files
- Consider using Celery for async image generation
- Use a database like PostgreSQL for production

### Frontend Optimizations (Already Implemented)

- ✅ Code splitting and lazy loading
- ✅ Image lazy loading
- ✅ React Query for data caching
- ✅ Optimized bundle with Vite
- ✅ Tree shaking for minimal bundle size
- ✅ Memoization to prevent unnecessary re-renders

### Infrastructure

- Use a CDN for serving static assets
- Enable HTTP/2 or HTTP/3
- Enable gzip/brotli compression
- Set appropriate cache headers
- Consider using a queue system (Redis + Celery) for long-running tasks

## Monitoring

Consider adding:
- Error tracking (Sentry)
- Performance monitoring (New Relic, DataDog)
- Analytics (Google Analytics, Plausible)
- Uptime monitoring

## Security Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Set `DEBUG = False`
- [ ] Configure `ALLOWED_HOSTS` properly
- [ ] Set up HTTPS/TLS
- [ ] Configure CORS appropriately
- [ ] Enable CSRF protection
- [ ] Set security headers (CSP, HSTS, etc.)
- [ ] Regular dependency updates
- [ ] Input validation and sanitization
- [ ] Rate limiting on API endpoints

## Troubleshooting

### CORS Issues

If you encounter CORS errors, ensure:
1. `django-cors-headers` is installed
2. `CORS_ALLOWED_ORIGINS` includes your frontend URL
3. Middleware is properly configured

### Image Generation Timeout

If generation times out:
1. Increase timeout in Vite config (already set to 120s)
2. Consider using Celery for async processing
3. Optimize model parameters (reduce steps, use speed preset)

### Build Errors

If frontend build fails:
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Clear build cache: `rm -rf dist`
3. Check Node.js version (18+ required)

## Scaling

For high traffic:
1. Use a load balancer
2. Deploy multiple backend instances
3. Use a CDN for frontend
4. Implement caching strategy
5. Consider GPU sharing for multiple workers
6. Use a message queue for job distribution