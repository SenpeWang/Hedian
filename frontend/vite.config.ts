import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端地址：dev 代理用。默认本机 5002；部署到其他机器时通过环境变量 VITE_BACKEND 覆盖。
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backend = env.VITE_BACKEND || 'http://localhost:5002'

  return {
    plugins: [vue()],
    server: {
      proxy: {
        '/start': backend,
        '/data': backend,
        '/audio': backend,
        '/status': backend,
        '/api': backend,
      },
    },
    build: {
      outDir: 'dist',
      emptyOutDir: false,
    },
  }
})
