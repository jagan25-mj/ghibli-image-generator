# Ghibli Image Generator - Frontend

Modern React.js frontend for the Ghibli Image Generator, built with performance and UX in mind.

## Features

- ⚡️ **Vite** - Lightning fast HMR and build times
- ⚛️ **React 18** - Latest React features
- 🎨 **Tailwind CSS** - Utility-first CSS framework
- 📦 **Code Splitting** - Optimized bundle size
- 🔄 **React Query** - Efficient data fetching and caching
- 📝 **React Hook Form** - Performant form handling
- 🎭 **Framer Motion** - Smooth animations
- 🍞 **React Hot Toast** - Beautiful notifications
- ♿️ **Accessible** - WCAG compliant components

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   ├── hooks/           # Custom React hooks
│   ├── services/        # API services
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── public/              # Static assets
└── index.html           # HTML template
```

## Performance Optimizations

- **Code Splitting**: Components are lazy-loaded for optimal bundle size
- **Image Lazy Loading**: Images load only when needed
- **React Query Caching**: API responses are cached and deduplicated
- **Memoization**: Components and callbacks are memoized to prevent unnecessary re-renders
- **Error Boundaries**: Graceful error handling prevents app crashes

## API Integration

The frontend communicates with the Django backend via REST API:

- `GET /api/config/` - Fetch available configuration options
- `POST /api/generate/` - Generate images with provided parameters

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT