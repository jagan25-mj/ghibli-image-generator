/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
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