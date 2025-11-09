import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import process from 'process';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Ensure Vite listens on all interfaces
    port: 5173,  
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://localhost:8000/', // Uses backend inside Docker
        changeOrigin: true,
        secure: false,
      },
    },
  },
});