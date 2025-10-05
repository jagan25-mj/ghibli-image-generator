import { memo } from 'react';

const Footer = memo(() => {
  return (
    <footer className="w-full max-w-7xl mx-auto px-4 sm:px-6 pb-8 text-center text-slate-400 text-xs">
      Made for fun • Not affiliated with Studio Ghibli • Images run locally
    </footer>
  );
});

Footer.displayName = 'Footer';

export default Footer;