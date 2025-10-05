import { memo } from 'react';

const Header = memo(() => {
  return (
    <header className="w-full max-w-7xl mx-auto px-4 sm:px-6 pt-6">
      <div className="glass">
        <div className="glass-header">
          <div className="flex items-center gap-2">
            <span className="dot bg-red-500"></span>
            <span className="dot bg-amber-500"></span>
            <span className="dot bg-green-500"></span>
          </div>
          <h1 className="font-bold text-base sm:text-lg">Ghibli Image Generator</h1>
          <div className="text-xs text-slate-300 hidden sm:block">Model: SD 1.5</div>
        </div>
        <div className="brandline"></div>
      </div>
    </header>
  );
});

Header.displayName = 'Header';

export default Header;