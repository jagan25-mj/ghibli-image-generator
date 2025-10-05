const LoadingSpinner = ({ size = 'large', text = 'Loading...' }) => {
  const sizeClasses = {
    small: 'w-6 h-6 border-2',
    medium: 'w-10 h-10 border-3',
    large: 'w-16 h-16 border-4',
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div
        className={`${sizeClasses[size]} border-white/25 border-t-white rounded-full animate-spin`}
        role="status"
        aria-label="Loading"
      />
      {text && <p className="mt-4 text-muted text-sm">{text}</p>}
    </div>
  );
};

export default LoadingSpinner;