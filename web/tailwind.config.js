/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: { extend: { boxShadow: { soft: '0 8px 30px rgba(15,23,42,.06)' } } },
  plugins: [],
};
