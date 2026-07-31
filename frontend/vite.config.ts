import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/start': 'http://10.152.88.66:5002',
      '/data': 'http://10.152.88.66:5002',
      '/audio': 'http://10.152.88.66:5002',
      '/status': 'http://10.152.88.66:5002',
      '/api': 'http://10.152.88.66:5002',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
