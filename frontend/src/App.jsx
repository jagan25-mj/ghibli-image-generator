import { Suspense, lazy } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Footer from './components/Footer';
import ErrorBoundary from './components/ErrorBoundary';
import LoadingSpinner from './components/LoadingSpinner';

// Lazy load the main generator component for code splitting
const ImageGenerator = lazy(() => import('./components/ImageGenerator'));

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen flex flex-col">
        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'rgba(20, 23, 31, 0.95)',
              color: '#e7edf6',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
              backdropFilter: 'blur(18px)',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#e7edf6',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#e7edf6',
              },
            },
          }}
        />

        {/* Header */}
        <Header />

        {/* Main Content */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
          <Suspense fallback={<LoadingSpinner />}>
            <ImageGenerator />
          </Suspense>
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </ErrorBoundary>
  );
}

export default App;