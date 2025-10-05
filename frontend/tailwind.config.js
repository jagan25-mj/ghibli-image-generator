/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0b0e13',
        panel: 'rgba(20, 23, 31, 0.7)',
        stroke: 'rgba(255, 255, 255, 0.1)',
        muted: '#9aa4b2',
        text: '#e7edf6',
      },
      animation: {
        'flow': 'flow 6s linear infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      keyframes: {
        flow: {
          '0%': { backgroundPosition: '0%' },
          '100%': { backgroundPosition: '300%' },
        },
      },
    },
  },
  plugins: [],
}