import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/analyze-ui': {
        target: 'http://0.0.0.0:8000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
